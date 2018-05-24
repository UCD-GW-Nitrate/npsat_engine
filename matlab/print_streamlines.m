function print_streamlines( S, sid )
%print_streamlines prints the streamlines of the structure S with id in sid
%   S The structure that is created by the program
%   sid a list of ids with the streamlines to be printed

hold on
for ii = 1; length(sid)
    Npnts = length(S(sid(ii),1).P);
    XYZ = nan(Npnts, 3);
    for jj = 1:Npnts
        XYZ(jj,:) = S(sid(ii),1).P(jj,1).XYZ;
    end
    
    plot3(XYZ(:,1), XYZ(:,2), XYZ(:,3),'.-')

end
hold off

