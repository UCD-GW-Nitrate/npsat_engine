#!/usr/bin/env python3
"""Capture fixed/moving mesh flow baselines; compare decoded VTU data, not timestamps.

Run from the repository root. Requires Python 3 and an MPI launcher, no Python packages.
Outputs are deliberately outside the source fixtures. Never overwrites an existing run.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
import zlib


def array_values(node):
    text = ''.join(node.itertext()).strip()
    if node.get('format') == 'ascii':
        return [float(x) for x in text.split()]
    # deal.II 9.3 inline compressed VTU: separately base64-encoded header/payload.
    encoded = ''.join(text.split())
    blocks = struct.unpack('<I', base64.b64decode(encoded[:8])[:4])[0]
    header_chars = ((4 * (3 + blocks) + 2) // 3) * 4
    header = struct.unpack('<' + 'I' * (3 + blocks),
                           base64.b64decode(encoded[:header_chars]))
    payload = base64.b64decode(encoded[header_chars:])
    raw, offset = b'', 0
    for size in header[3:]:
        raw += zlib.decompress(payload[offset:offset + size])
        offset += size
    assert offset == len(payload), 'Unexpected compressed payload size'
    assert len(raw) == (blocks - 1) * header[1] + header[2]
    kind = {'Float32': 'f', 'Float64': 'd', 'Int32': 'i', 'UInt8': 'B'}[node.get('type')]
    return list(struct.unpack('<' + kind * (len(raw) // struct.calcsize(kind)), raw))


def summarize(folder):
    result = {}
    for path in sorted(folder.glob('*.vtu')):
        root = ET.parse(str(path)).getroot()
        assert root.get('byte_order') == 'LittleEndian'
        assert root.get('compressor') == 'vtkZLibDataCompressor'
        piece = root.find('./UnstructuredGrid/Piece')
        arrays = {}
        for group in piece:
            for i, node in enumerate(group.findall('DataArray')):
                name = group.tag + '/' + node.get('Name', str(i))
                values = array_values(node)
                assert all(math.isfinite(v) for v in values), name
                arrays[name] = values
        result[path.name] = arrays
    assert result, 'No VTU output produced'
    return result


def compare(actual, expected):
    assert actual.keys() == expected.keys(), 'Different output files'
    worst = 0.0
    for filename, arrays in actual.items():
        assert arrays.keys() == expected[filename].keys()
        for name, values in arrays.items():
            old = expected[filename][name]
            assert len(values) == len(old), (filename, name, 'size changed')
            for a, b in zip(values, old):
                delta = abs(a - b)
                worst = max(worst, delta)
                tolerance = 0 if name.startswith('Cells/') else 2e-6 * max(1, abs(b))
                assert delta <= tolerance, (filename, name, a, b)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--compare', help='Previous summary.json from the same MPI rank count')
    args = parser.parse_args()
    assert args.ranks > 0
    exe = Path(args.exe).resolve()
    dest = Path(args.output).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    source = Path('examples/interpolation_v2').resolve()
    inputs = dest / 'input'
    shutil.copytree(str(source), str(inputs))
    # Uniform K=1, bottom h=10, recharge=0.001 on a 100x100 box.
    # Exact fixed-mesh solution h(z)=10+0.001*z. Moving top starts at 12.
    (inputs / 'top.values').write_text('12 12\n')
    (inputs / 'conductivity.values').write_text('1 1\n')
    (inputs / 'recharge.values').write_text('0.001 0.001\n')
    (inputs / 'dirichlet.dat').write_text('1\nBOT 4 constant.master\n0 0\n100 0\n100 100\n0 100\n')
    template = (source / 'smoke.prm').read_text()
    results = {'ranks': args.ranks, 'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
               'cases': {}, 'diagnostics': {}}
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    for case, iterations in [('fixed', 1), ('moving', 3)]:
        folder = dest / case
        folder.mkdir()
        config = template.replace('examples/interpolation_v2/', str(inputs) + '/')
        config = config.replace('build-topo3/output/', str(folder) + '/')
        config = config.replace('set a Nonlinear iterations = 2',
                                'set a Nonlinear iterations = ' + str(iterations))
        config = config.replace('set a Prefix = smoke', 'set a Prefix = flow')
        prm = folder / 'case.prm'
        prm.write_text(config)
        with (folder / 'run.log').open('w') as log:
            subprocess.run(['mpirun', '-np', str(args.ranks), str(exe), '-p', str(prm)],
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=180)
        log = (folder / 'run.log').read_text()
        assert log.count('Solved in ') == iterations, 'Missing solves: inspect ' + str(folder)
        data = summarize(folder)
        assert len(data) == iterations * args.ranks
        # Verify physics, not only agreement with the old executable (VTU is Float32).
        for arrays in data.values():
            xyz = arrays['Points/0']
            heads = arrays['PointData/Head']
            assert len(xyz) == 3 * len(heads)
            assert all(abs(h - (10 + 0.001 * xyz[3*i+2])) < 3e-6
                       for i, h in enumerate(heads)), 'Analytic head check failed'
        tops = [max(v['Points/0'][2::3]) for k, v in data.items() if k.startswith('flow000.')]
        assert abs(max(tops) - 12) < 1e-6
        if case == 'moving':
            final_tops = [max(v['Points/0'][2::3]) for k, v in data.items() if k.startswith('flow002.')]
            assert 10 < max(final_tops) < 10.02, 'Mesh did not move as expected'
        recharge = [float(x) for x in re.findall(r'QRCH: \[([^\]]+)\]', log)]
        assert len(recharge) == iterations and all(abs(q - 10) < 1e-8 for q in recharge)
        results['cases'][case] = data
        results['diagnostics'][case] = {
            'recharge': recharge,
            'solver_iterations': re.findall(r'Solved in (\d+) iterations', log),
            'solver_residuals': [list(match) for match in
                                 re.findall(r'Iter:(\d+): ([^\s]+)', log)]}
    if args.compare:
        old = json.loads(Path(args.compare).read_text())
        assert old['ranks'] == args.ranks
        results['comparison_max_absolute_difference'] = {
            name: compare(data, old['cases'][name]) for name, data in results['cases'].items()}
        assert results['diagnostics'] == old['diagnostics'], 'Solver/source diagnostics changed'
    (dest / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k: v for k, v in results.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
