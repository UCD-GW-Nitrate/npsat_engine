function Wells = pick_plot( Nwells )

hold on
Wells = nan(Nwells,3);
for ii = 1:Nwells
    t = ginput(2);
    Wells(ii,:) = [t(1,1) t(1,2) t(2,2)];
    plot([Wells(ii,1) Wells(ii,1)], [Wells(ii,2) Wells(ii,3)])
end

hold off

