function wells = readWells(filename)

fid = fopen(filename,'r');
Nwells = fscanf(fid, '%d',1);
temp = fscanf(fid, '%f',Nwells*5);
fclose(fid);
wells = reshape(temp, 5, Nwells)';

