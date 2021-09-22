function [p, MSH] = readOBJmesh(varargin)
% [p, MSH] = readOBJmesh('filename');
%
% Input 
%       readOBJmesh('filename')
%       readOBJmesh('filename', Nsides)
%       readOBJmesh('filename', Nsides, Nalloc)
%       readOBJmesh('filename', Nsides, Np_alloc, Nel_alloc)
%
% readOBJmesh reads the vertices (v), faces (f) and lines (l) from an obj file.
% All other info is ignored.
% The programs that write obj files can write into the file any kind of
% geometry. FOr example if there are degenerated polygons these will be
% printed. Therefore before attempting to use this script make sure that
% the onj file is somewhat clean.
% Just by reading a file does not mean that the mesh would make any sense.
% if the obj file represent something that it's not a mesh then the result
% wont make any sense.

if nargin == 1
    filename = varargin{1};
    Nsides = 3;
    Np = 10000;
    Nel = 10000;
elseif nargin == 2
    filename = varargin{1};
    Nsides = varargin{2};
    Np = 10000;
    Nel = 10000;
elseif nargin == 2
    filename = varargin{1};
    Nsides = varargin{2};
    Np = varargin{3};
    Nel = varargin{3};
elseif nargin == 3
    filename = varargin{1};
    Nsides = varargin{2};
    Np = varargin{3};
    Nel = varargin{4};
end


fid = fopen(filename, 'r');

p = nan(Np, 3);
MSH = nan(Nel, Nsides);
cnt_p = 1;
cnt_el = 1;
while ~feof(fid)
    temp = fgetl(fid);
    if isempty(temp)
        continue;
    end
    if strcmp(temp(1),'v')
        p(cnt_p,:) = cell2mat(textscan(temp,'v %f %f %f'));
        cnt_p = cnt_p + 1;
    end
    if strcmp(temp(1),'f') || strcmp(temp(1),'l')
        temp1=temp(2:end);
        tmp = cell2mat(textscan(temp1,' %f'))';
        MSH(cnt_el,1:length(tmp)) = tmp;
        cnt_el = cnt_el + 1;
    end
    
end
p(cnt_p:end,:) = [];
MSH(cnt_el:end,:) = [];
fclose(fid);

