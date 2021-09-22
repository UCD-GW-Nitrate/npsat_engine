function [dofs, simTime] = readNPSATLogStat(filename)
fid = fopen(filename, 'r');
dofs = [];
simTime = [];
while ~feof(fid)
    tline = fgetl(fid);
    if strcmp(tline, '	Setting up system...')
        tline = fgetl(fid);
        c = textscan(tline,'%s %s %s %s %s %f');
        dofs = [dofs; c{1,6}];
    end
    if strcmp(tline, '+---------------------------------------------+------------+------------+')
        for ii = 1:5
            tline = fgetl(fid);
        end
        c = textscan(tline,'%s %s %s %f %s %fs %s %f%s');
        assemble = c{1,6};
        tline = fgetl(fid);
        c = textscan(tline,'%s %s %s %f %s %fs %s %f%s');
        output = c{1,6};
        tline = fgetl(fid);
        c = textscan(tline,'%s %s %s %f %s %fs %s %f%s');
        setup = c{1,6};
        tline = fgetl(fid);
        c = textscan(tline,'%s %s %s %f %s %fs %s %f%s');
        solve = c{1,6};
        simTime = [simTime; setup assemble solve output];
    end
end

fclose(fid);
end

