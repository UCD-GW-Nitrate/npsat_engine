#!/usr/bin/env python3
"""Portable INI, shared-data and MPI failure regression suite (Python 3 stdlib)."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_baseline import summarize, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reader')
    parser.add_argument('--reference-reader')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    exe = str(Path(args.exe).resolve())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    example = repo / 'examples/shared_flow'
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    results = []

    def run_case(name, fixture='fixed.ini', ranks=2, replace=None, files=None,
                 validate=False, expected_error=None):
        folder = output / name
        folder.mkdir()
        shutil.copytree(str(example / 'input'), str(folder / 'input'))
        (folder / 'output').mkdir()
        config = (example / fixture).read_text().replace('Main = examples/shared_flow', 'Main = ' + str(folder))
        for old, new in (replace or []):
            assert old in config, old
            config = config.replace(old, new)
        for filename, contents in (files or {}).items():
            target = folder / 'input' / filename
            if contents is None:
                target.unlink()
            else:
                target.write_text(contents)
        config_path = folder / 'case.ini'
        config_path.write_text(config)
        command = ['mpirun', '-np', str(ranks), exe, '-c', str(config_path), '--print-config']
        if validate:
            command.append('--validate-only')
        completed = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True, timeout=90)
        (folder / 'run.log').write_text(completed.stdout)
        if expected_error:
            assert completed.returncode != 0, name + ' unexpectedly succeeded'
            assert expected_error in completed.stdout, name + ': wrong failure, see run.log'
            assert not list((folder / 'output').glob('*.vtu')), name + ' wrote a solution'
        else:
            assert completed.returncode == 0, name + ' failed, see run.log'
            assert 'INPUT_VALIDATION_OK' in completed.stdout
        results.append({'case': name, 'ranks': ranks, 'exit_code': completed.returncode,
                        'expected_error': expected_error})
        return folder, completed.stdout

    for ranks in (1, 2):
        for fixture in ('fixed.ini', 'scenario1.ini', 'moving.ini', 'sources.ini'):
            name = fixture[:-4] + '-mpi' + str(ranks)
            folder, log = run_case(name, fixture, ranks)
            data = summarize(folder / 'output')
            applied_recharge = [float(x) for x in re.findall(r'QRCH: \[([^\]]+)\]', log)]
            assert applied_recharge and all(abs(q-(40 if fixture == 'scenario1.ini' else 10)) < 1e-9
                                            for q in applied_recharge), name
            if fixture != 'sources.ini':
                base, gradient, top = (20, .002, 24) if fixture == 'scenario1.ini' else (10, .001, 12)
                for arrays in data.values():
                    xyz, head = arrays['Points/0'], arrays['PointData/Head']
                    assert all(abs(h - base - gradient*xyz[3*i+2]) < 3e-6 for i,h in enumerate(head)), name
                first_top = max(max(a['Points/0'][2::3]) for f,a in data.items() if '000.' in f)
                assert abs(first_top-top) < 1e-6
                if fixture in ('fixed.ini', 'moving.ini'):
                    golden = json.loads((repo/'tests/baselines/step0'/('serial.json' if ranks == 1 else 'mpi2.json')).read_text())
                    prefix = 'shared' if fixture == 'fixed.ini' else 'moving'
                    normalized = {f.replace(prefix, 'flow', 1):a for f,a in data.items()}
                    difference = compare(normalized, golden['cases']['fixed' if fixture == 'fixed.ini' else 'moving'])
                    results[-1]['baseline_max_difference'] = difference
            else:
                assert 'Selected.Well=17,0,-0.25' in log
                assert 'Selected.Stream=0,0,0.0002' in log
                assert abs(float(re.search(r'QWELLS: \[([^\]]+)\]',log).group(1)) + .25) < 1e-10
                assert abs(float(re.search(r'Q_streams = (\S+)',log).group(1)) - .32) < 1e-10

    # Positive validation-only cases: no output, single-column broadcast,
    # a different stress, and MNW parsing without pretending MNW assembly exists.
    run_case('validate-mnw', 'sources.ini', replace=[('WellModel = legacy','WellModel = mnw')], validate=True)
    folder, log = run_case('validate-source-step1', 'sources.ini',
                          replace=[('Start_step = 0','Start_step = 1')], validate=True)
    assert 'Selected.Well=17,0,-0.5' in log
    rate_line = next(line for line in log.splitlines() if line.startswith('Selected.Stream='))
    assert abs(float(rate_line.rsplit(',',1)[1]) - .0006) < 1e-15
    run_case('default-vertical', replace=[('Vertical = 0,1','Vertical =')])
    run_case('ky-only', replace=[('Porosity = porosity.master','Porosity = porosity.master\nKy = conductivity.master')])
    run_case('missing-output', replace=[('Output = output','Output = missing-output')],
             expected_error='Output directory does not exist')
    bad_cases = [
        ('unknown-key', [('System_tol = 1e-8','System_typo = 1e-8')], {}, 'unrecognised option'),
        ('negative-step', [('Start_step = 0','Start_step = -1')], {}, 'must be nonnegative'),
        ('missing-column', [('Start_step = 0','Start_step = 2')], {}, 'Unavailable stress column'),
        ('transient', [('Steady_state = 1','Steady_state = 0')], {}, 'only supports'),
        ('bad-nxyz', [('Nxyz = 2,2,1','Nxyz = 2,0,1')], {}, 'cell counts'),
        ('bad-vertical', [('Vertical = 0,1','Vertical = 0,0.8,0.2,1')], {}, 'increase strictly'),
        ('missing-values', [], {'recharge.values':None}, 'Cannot open file'),
        ('ragged-values', [], {'recharge.values':'1 2\n3\n'}, 'Inconsistent column count'),
        ('garbage-values', [], {'recharge.values':'1 2 junk\n'}, 'Invalid numeric token'),
        ('nonfinite-values', [], {'recharge.values':'1 nan\n'}, 'Invalid numeric token'),
        ('overflow-token', [], {'recharge.values':'1 1e99999\n'}, 'Invalid numeric token'),
        ('empty-values', [], {'recharge.values':''}, 'Empty numeric table'),
        ('invalid-master', [], {'top.master':'-1\n'}, 'Invalid region count'),
        ('short-hidden-region', [('Start_step = 0','Start_step = 2')],
         {'top.values':'12 24 36\n',
          'top.master':'2\n2 CONST unused top.values\n0 0\n100 100\n2 CONST unused constant.values\n0 0\n100 100\n'},
         'Unavailable stress column'),
        ('invalid-boundary', [], {'dirichlet.dat':'-1\n'}, 'Invalid Dirichlet file'),
        ('missing-boundary', [('Dirichlet = dirichlet.dat','Dirichlet = absent.dat')], {}, 'Cannot open file'),
        ('ghb-not-implemented', [('GHB =\n','GHB = ghb.dat\n')], {}, 'milestone 2'),
        ('bad-lateral-id', [('Rch_Data = recharge.master','Rch_Data = lateral.master')],
         {'lateral.spatial':'2 0\nLINEAR NEAREST\n0 0\n0 100\n0 999\n'}, 'Lateral value row ID'),
    ]
    for name, changes, files, error in bad_cases:
        run_case(name, replace=changes, files=files, validate=True, expected_error=error)
    for name, changes, files, error in [
        ('duplicate-well', [], {'wells.csv':(example/'input/wells.csv').read_text()+'17,30,30,9,1,0,0.1,0.2,1\n'}, 'Duplicate'),
        ('bad-well-row', [], {'wells.csv':(example/'input/wells.csv').read_text().replace('9,1,0,','9,1,5,')}, 'invalid zero-based rate row'),
        ('short-rate-series', [('Start_step = 0','Start_step = 2')], {}, 'unavailable stress column'),
        ('missing-rate-file', [], {'well_rates.dat':None}, 'Cannot open file'),
        ('bad-stream-row', [], {'streams.dat':'1\n2 8 10\n10 50\n90 50\n'}, 'invalid zero-based rate row'),
        ('mnw-not-implemented', [('WellModel = legacy','WellModel = mnw')], {}, 'MNW assembly is milestone 3')
    ]:
        run_case(name, 'sources.ini', replace=changes, files=files,
                 validate=name != 'mnw-not-implemented', expected_error=error)
    if args.reader:
        probes = [args.reader] + ([args.reference_reader] if args.reference_reader else [])
        texts = []
        for index, probe in enumerate(probes):
            text = subprocess.check_output([str(Path(probe).resolve()), str(example/'input')], env=env,
                                           universal_newlines=True, timeout=90)
            (output/('reader-'+str(index)+'.txt')).write_text(text)
            texts.append(text)
        assert texts[0] == (example/'expected_reader.txt').read_text(), 'Reader result differs from pinned trace_dbg reference'
        if len(texts) == 2:
            assert texts[0] == texts[1], 'v1/v2 reader results differ'
            results.append({'reader_comparison': 'identical'})
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print('Passed {} checks; artifacts: {}'.format(len(results), output))


if __name__ == '__main__':
    main()
