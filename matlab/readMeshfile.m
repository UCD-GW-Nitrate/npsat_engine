function [msh_nd, msh_el] = readMeshfile(filename)
% [msh_nd, msh_el] = readMeshfile(filename) reads the mesh file
% filename is the name of the file
%
% msh_nd: are the nodes of the mesh
%
% msh_el: are the elements. The ids may use 0 based numbering.
%         To fix this add 1 to the msh_el matrix

fid = fopen(filename,'r');
temp = fscanf(fid, '%d',2);
Nnd = temp(1);
Nel = temp(2);
temp = fscanf(fid, '%f',Nnd*3);
msh_nd = reshape(temp, 3, Nnd)';
temp = fscanf(fid, '%d',Nel*4);
msh_el = reshape(temp, 4, Nel)';



fclose(fid);