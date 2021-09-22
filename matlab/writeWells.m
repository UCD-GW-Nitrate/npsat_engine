function writeWells(filename,wells)

fid = fopen(filename,'w');

fprintf(fid,'%d\n', size(wells,1));
fprintf(fid, '%f %f %f %f %f\n', wells');
fclose(fid);