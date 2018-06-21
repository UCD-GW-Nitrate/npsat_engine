function writeMeshfile(filename, msh_nd, msh_el)
% The msh_el must be defined with 1 based numbering. The code will subtract
% one after finishes the orientation checks

for ii = 1:size(msh_el,1)
    if ~ispolycw(msh_nd(msh_el(ii,[1 2 3 4]),1), msh_nd(msh_el(ii,[1 2 3 4]),2))
        msh_el(ii,:) = msh_el(ii,[1 4 3 2]);
    end
end


fid = fopen(filename,'w');
fprintf(fid, '%d %d\n', [size(msh_nd,1) size(msh_el,1)]);
fprintf(fid, '%f %f %f\n', [msh_nd(:,1:2) zeros(size(msh_nd,1),1)]');
msh_el = msh_el-1;
fprintf(fid, '%d %d %d %d\n', msh_el');
fclose(fid);