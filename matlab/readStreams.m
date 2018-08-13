function STRM = readStreams(filename)
% STRM = readStreams(filename) reads the streams from the NPSAT input file
%
% filename is the name of the input stream file
%
% STRM:  is the output structure with the following fields
%      poly:    the coordinates of the polygon of the stream segment
%      Q:       the rate, which should be in [L/T]
%      area:    the area of the stream segment. This is not read but
%               calculated by the coordinates in poly


fid = fopen(filename,'r');
temp = fscanf(fid, '%d',1);
Nstrm = temp(1);
for ii = 1:Nstrm
    temp = textscan(fid, '%d %f',1);
    Np = temp{1,1}(1);
    q = temp{1,2}(1);
    poly = nan(Np,2);
    temp = textscan(fid, '%f %f',Np);
    poly(:,1) = temp{1,1};
    poly(:,2) = temp{1,2};
    %for jj = 1:Np   
    %    poly(jj,1) = temp(1);
    %    poly(jj,2) = temp(2);
    %end
    
    STRM(ii,1).poly = poly;
    STRM(ii,1).Q = q;
    STRM(ii,1).area = polyarea(poly(:,1), poly(:,2));
end

fclose(fid);

% ============ READ OLD FORMAT============== 
%fid = fopen(filename,'r');
%temp = fscanf(fid, '%d',1);
%Nstrm = temp(1);
%temp = fscanf(fid, '%f',Nstrm*6);
%STRM = reshape(temp, 6, Nstrm)';
%fclose(fid);