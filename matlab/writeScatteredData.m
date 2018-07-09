function writeScatteredData(filename, info, DATA)
% Writes scattered data into the appropriate format
% filename is the name of the file to be written
% info is a struct variable that requires 3 fields
%   PDIM: is the number of columns that correspond to coordinates
%           This is 1 for 1D points of 2 for 2D points.
%   TYPE: Valid options for type are FULL, HOR or VERT
%   MODE: Valid options for mode are SIMPLE or STRATIFIED
% DATA the data to be printed. Data should have as many columns as needed.
%       For example it can be :
%       [x v]
%       [x v1 z1 v2 z2 ... vn-1 zn-1 vn]
%       [x y v]
%       [x y v1 z1 v2 z2 ... vn-1 zn-1 vn]
%       .
%       .
%       .

fid = fopen(filename,'w');
% Write flags
fprintf(fid, 'SCATTERED\n');
fprintf(fid, [info.TYPE '\n']);
fprintf(fid, [info.MODE '\n']);

Ndata = size(DATA,2) - info.PDIM;
fprintf(fid, '%d %d\n', [size(DATA,1) Ndata]);
% configure format
frmt = [];
for ii = 1:info.PDIM
    frmt = [frmt '%f '];
end
for ii = 1:Ndata-1
    frmt = [frmt '%f '];
end
frmt = [frmt '%f\n'];
% print data
fprintf(fid, frmt, DATA');
fclose(fid);