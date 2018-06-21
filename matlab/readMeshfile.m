function [msh_nd, msh_el] = readMeshfile(filename)
fid = fopen(filename,'r');
temp = fscanf(fid, '%d',2);
Nnd = temp(1);
Nel = temp(2);
temp = fscanf(fid, '%f',Nnd*3);
msh_nd = reshape(temp, 3, Nnd)';
temp = fscanf(fid, '%d',Nel*4);
msh_el = reshape(temp, 4, Nel)';



fclose(fid);