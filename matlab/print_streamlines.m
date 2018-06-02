function print_streamlines( S, sid, bcells )
%print_streamlines prints the streamlines of the structure S with id in sid
%   S The structure that is created by the program
%   sid a list of ids with the streamlines to be printed
%   bcells is a boolean flag. Set it to true to plot the cells together
%   with the streamline

if isempty(sid)
    sid = 1:length(S);
end

    hold on
    for ii = 1:length(sid)
        Npnts = length(S(sid(ii),1).P);
        dim = length(S(sid(ii),1).P(1,1).XYZ);
        XYZ = nan(Npnts, dim);
        for jj = 1:Npnts
            XYZ(jj,:) = S(sid(ii),1).P(jj,1).XYZ;
        end
        if dim == 3
            plot3(XYZ(:,1), XYZ(:,2), XYZ(:,3),'.-')
        elseif dim == 2
            plot(XYZ(:,1), XYZ(:,2),'.-')
        end

        if bcells
            for kk = 1:length(S(sid(ii),1).Cells)
                if ~isempty(S(sid(ii),1).Cells(kk,1).Nodes)
                    plot_cell(S(sid(ii),1).Cells(kk,1).Nodes, S(sid(ii),1).Cells(kk,1).Type, dim)
                end

            end
        end

    end
    hold off
end




function plot_cell(P, type, dim)
    clr = [0.5 0.5 0.5];
    if type == 1
        clr(1) = 1;
    elseif type == 2
        clr(2) = 1;
    end
    
    if dim == 3
        % top and bottom face
        plot3(P([1:4 1],1), P([1:4 1],2), P([1:4 1],3),'color',clr);
        plot3(P([5:8 5],1), P([5:8 5],2), P([5:8 5],3),'color',clr);

        % vertical lines
        plot3(P([1 5],1), P([1 5],2), P([1 5],3),'color',clr);
        plot3(P([2 6],1), P([2 6],2), P([2 6],3),'color',clr);
        plot3(P([3 7],1), P([3 7],2), P([3 7],3),'color',clr);
        plot3(P([4 8],1), P([4 8],2), P([4 8],3),'color',clr);
    elseif dim == 2
        plot(P([1:4 1],1), P([1:4 1],2),'color',clr);
    end
end

