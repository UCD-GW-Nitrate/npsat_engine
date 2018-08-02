function STRM = readStreams(filename)
fid = fopen(filename,'r');
temp = fscanf(fid, '%d',1);
Nstrm = temp(1);
temp = fscanf(fid, '%f',Nstrm*6);
STRM = reshape(temp, 6, Nstrm)';
fclose(fid);