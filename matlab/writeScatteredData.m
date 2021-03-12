function writeScatteredData(filename, TYPE, COORDS, DATA, TR)
%% writeScatteredData(filename, info, DATA, TR)
%
% Writes scattered data into the appropriate format
% filename: is the name of the file to be written
% TYPE: This is the interpolation method. it can be LINEAR or NEAREST.
% However for the 3D interpolation both the horizontal and vertical type
% must be defined.
% COORDS: [x] or [x y]
%
% DATA: the data to be printed. Data should have as many columns as needed.
%       For 2D the data should have 0ne column.
%       For 3D with NEAREST vertical TYPE{2} then the number of data must be
%       odd. [value Elevatio value Elevation .... Value]
%       For 3D with LINEAR vertical then the number of data must be even.
%       [value Elevatio value Elevation .... Value Elevation]
%       
%       For example it can be :
%       [v]
%       [v1 z1 v2 z2 ... vn-1 zn-1 vn]
%       [v1 z1 v2 z2 ... vn-1 zn-1 vn]
%       .
%       .
%       .
% TR: Triangulation data
%    [id1 id2 i3]
%       .
%       .
%       .
% The ids must start on 1. The function will subtract one during printing
%
% Examples
% For 2D interpolation such as top, bottom elevation or recharge
% writeScatteredData(filename, struct('DIM', '2D', 'TYPE', {'LINEAR'}), DATA)


fid = fopen(filename,'w');
% Write flags
fprintf(fid, 'SCATTERED\n');
if length(TYPE) == 1
    fprintf(fid, '2D\n');
    fprintf(fid, [TYPE{1} '\n']);
elseif length(TYPE) == 2
    fprintf(fid, '3D\n');
    fprintf(fid, [TYPE{1} ' ' TYPE{2} '\n']);
end

fprintf(fid, '%d %d %d\n', [size(COORDS,1) size(DATA,2), size(TR,1)]);

fprintf(fid, [repmat('%f ', 1, size(COORDS,2) + size(DATA,2)) '\n'], [COORDS DATA]');
fprintf(fid, '%d %d %d\n',[TR-1]');
fclose(fid);