function out = read_Scattered( filename, PDIM)
% out = read_Scattered reads the scattered interpolation file
% filename is the name of the file
%
% PDIM is the dimension of the points. This can be 1 or 2.
%
% out is a structure with the following fields
%   p the interpolation points
%   v the values
%   info information about the interpolation data

fid = fopen(filename);
interp_type = fgetl(fid);
if strcmp(interp_type, 'SCATTERED')
    out.info.TYPE = fgetl(fid);
    out.info.MODE = fgetl(fid);
    N = fscanf(fid, '%d %d\n', 2);
    
    frmt ='%f';
    for ii = 1:PDIM-1
        frmt =[frmt ' %f']; 
    end

    for ii = 1:N(2)
        frmt = [frmt ' %f'];
    end
    frmt = [frmt '\n'];
    out.p = nan(N(1), PDIM);
    out.v = nan(N(1), N(2));
    
    for ii = 1:N(1)
        temp = fscanf(fid, frmt, N(2)+PDIM);
        out.p(ii,:) = [temp(1:PDIM)'];
        ll = [];
        for jj = 1:N(2)
            ll = [ll temp(jj+2)]; 
        end
        out.v(ii,:) = ll;
    end
    
else
    out = [];
end

fclose(fid);



