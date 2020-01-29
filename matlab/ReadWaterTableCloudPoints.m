function WTC = ReadWaterTableCloudPoints( inputStruct )
% WTC = ReadWaterTableCloudPoints( inputStruct)
%
% Examples:
% WTC = ReadWaterTableCloudPoints( struct('prefix','Testprefix', 'nproc',4, 'iter', 6, 'print_file', true))
% 
% ReadWaterTableCloudPoints Reads the point cloud from different processors
% The input is a structure with the following fields.
% Inside the brackets is shown the default value. If the default value is
% good then it can be omitted from the structure. The prefix field is the
% only one that is required.
%   prefix [required]: the prefix of the file name. This should be the Prefix as 
%                      defined in section J.a of the parameter file. This
%                      is required.
%                      The name that the function will try to read is 
%                      [prefix suffix '_' num2str(iter,'%03d') '_' num2str(iproc,'%04d') '.xyz']
%   suffix [top]:      this is a suffix that by default the program appends 
%                      the word top, and as for now the user has no option to choose something different
%   nproc[1]:          Number of processors used in the simulation.
%   iter[1]:           The iteration to read. Note that iter = 1 refers to the first iteration which has id 000 
%   print_file[false]: if print_file is true it will print all the unique
%                      points from all processors into the file [prefix suffix '_ALLproc_iter.xyz']     

if isfield(inputStruct,'prefix')
    prefix = inputStruct.prefix;
else
    error('The function arguments have changed and a ''prefix'' field is required')
end

if isfield(inputStruct,'suffix')
    suffix = inputStruct.suffix;
else
    suffix = 'top';
end

if isfield(inputStruct,'nproc')
    nproc = inputStruct.nproc;
else
    nproc = 1;
end

if isfield(inputStruct,'iter')
    iter = inputStruct.iter-1;
else
    iter = 0;
end

if isfield(inputStruct,'print_file')
    print_file = inputStruct.print_file;
else
    print_file = false;
end




Elev_new = [];
for ii = 0:nproc-1
    fid = fopen([prefix suffix '_' num2str(iter,'%03d') '_' num2str(ii,'%04d') '.xyz'],'r');
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
    fid = fopen([prefix suffix '_ALLproc_' num2str(iter,'%03d') '.xyz'], 'w');
    fprintf(fid, '%d\n', size(WTC,1));
    fprintf(fid, '%f %f %f\n', WTC');
    fclose(fid);
end


end

