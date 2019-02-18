function wells = readWells(filename)
% wells = readWells(filename)
% Reads the well input file. 
% The output is a matric Nwellsx5 where the rows are:
% [X Y top bottom Q]

fid = fopen(filename,'r');
Nwells = fscanf(fid, '%d',1);
temp = fscanf(fid, '%f',Nwells*5);
fclose(fid);
wells = reshape(temp, 5, Nwells)';

