function out = read_Scattered( filename )
%read_Scattered reads the scattered interpolation file
%   No further explanation

fid = fopen(filename);
interp_type = fgetl(fid);
if strcmp(interp_type, 'SCATTERED')
    N = fscanf(fid, '%d %d\n', 2);
    frmt ='%f %f';
    for ii = 1:N(2)
        frmt = [frmt ' %f'];
    end
    frmt = [frmt '\n'];
    out.p = nan(N(1),2);
    out.v = nan(N(1), N(2));
    
    for ii = 1:N(1)
        temp = fscanf(fid, frmt, N(2)+2);
        out.p(ii,:) = [temp(1) temp(2)];
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



