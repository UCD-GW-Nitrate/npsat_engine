function out = read_Scattered( filename, PDIM)
% out = read_Scattered( filename, PDIM) reads the scattered interpolation file
% filename: is the name of the file
%
% PDIM: is the dimension of the points. This can be 1 or 2.
%
% out: is a structure with the following fields:
%     p: the interpolation points
%     v: the values
%     info: information about the interpolation data

fid = fopen(filename);
interp_type = fgetl(fid);
if strcmp(interp_type, 'SCATTERED')
    out.info.TYPE = fgetl(fid);
    out.info.MODE = fgetl(fid);
    N = fscanf(fid, '%d %d %d\n', 3);
    
    out.Data = cell2mat(textscan(fid, repmat('%f ',1, PDIM + N(2)), N(1)));
    out.TR = cell2mat(textscan(fid, '%f %f %f', N(3))) + 1;
else
    out = [];
end

fclose(fid);



