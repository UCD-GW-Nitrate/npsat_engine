function WTC = ReadWaterTableCloudPoints( prefix, nproc, print_file, iter )
% WTC = ReadWaterTableCloudPoints( prefix, nproc, print_file )
%
% ReadWaterTableCloudPoints Reads the point cloud from different processors
%   prefix: the prefix of the file name. The name that the function will
%           try to read is prefix [prefix num2str(iproc,'%04d') '.xyz']
%   nproc:  Number of processors used in the simulation
%   print_file: if print_file is true will print all the unique points in
%   the file [prefix 'ALL.xyz'] 



Elev_new = [];
for ii = 0:nproc-1
    fid = fopen([prefix num2str(ii,'%04d') '.xyz'],'r');
    Np = fscanf(fid, '%d',1);
    temp = fscanf(fid, '%f',Np*4);
    fclose(fid);
    temp = reshape(temp, 4, Np)';
    Elev_new = [Elev_new;temp];
end

% The following it is used to remove the duplicate points
FnewElev = scatteredInterpolant(Elev_new(:,1), Elev_new(:,2), Elev_new(:,4));
WTC = [FnewElev.Points FnewElev.Values];

if print_file
    fid = fopen([prefix 'iter_' num2str(iter) '_ALL.xyz'], 'w');
    fprintf(fid, '%d\n', size(WTC,1));
    fprintf(fid, '%f %f %f\n', WTC');
    fclose(fid);
    
end


end

