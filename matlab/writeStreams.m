function writeStreams(filename, Streams)
% writeStreams(filename, Streams) 
% Writes stream data to a file
%
% filename: is the name of the file
% 
% Streams: is a structure with the following fields
%   poly: describes a polygon or line entity by its vertices in the form [X Y]. 
%         The line entity is defined by 2 points whereas the polygon entity
%         mus be 4 vertices. If a line entity is given then the widths
%         neccesary in the structure.
%   Q:    is the pumping rate associated with the entity. 
%         This can be either a single number or the name of a file that 
%         contains data about the rate. However the later option is not
%         present in the model yet.
%   w:    The width of the stream for the entities that are lines


fid = fopen(filename,'w');
fprintf(fid, '%d\n', length(Streams));
for ii = 1:length(Streams)
    nVerts = size(Streams(ii,1).poly,1);
    fprintf(fid, '%d', nVerts);
    fprintf(fid, '  %s', num2str(Streams(ii,1).Q));
    if nVerts == 2
        fprintf(fid, ' %d\n', Streams(ii,1).w);
    else
        fprintf(fid, '\n');
    end
    fprintf(fid, '%0.3f %0.3f\n', Streams(ii,1).poly');
end
fclose(fid);
