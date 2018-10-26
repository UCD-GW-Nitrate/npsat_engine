function plotURFs( URFS, ids)
% plotURFs( URFS, ids ) Plots the unit response functions
%   URFS: is a structure that is the output of the readURFS
%   ids: are the entity ids. AN entity can be a well a stream or something
%        that is used to group streamlines

clf
hold on

if length(ids) == 1
    rows = find([URFS.Eid]' == ids);
    for jj = 1:length(rows)
        plot(1:length(URFS(rows(jj),1).URF), URFS(rows(jj),1).URF);
    end
    
else
    for ii = 1:length(ids)
        clr = rand(1,3);
        rows = find([URFS.Eid]' == ids(ii));

        for jj = 1:length(rows)
            plot(1:length(URFS(rows(jj),1).URF), URFS(rows(jj),1).URF, 'Color',clr);
        end
    end
    
end



end

