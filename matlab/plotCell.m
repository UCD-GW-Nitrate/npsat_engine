function plotCell(c, clr, lt,p)
%plotCell Summary of this function goes here
%   Detailed explanation goes here

hold on


if p == 1
    patch(c([1 2 3 4 1],1), c([1 2 3 4 1],2), c([1 2 3 4 1],3), clr, 'FaceAlpha',.3)
    patch(c([5 6 7 8 5],1), c([5 6 7 8 5],2), c([5 6 7 8 5],3), clr, 'FaceAlpha',.3)
    patch(c([1 2 6 5 1],1), c([1 2 6 5 1],2), c([1 2 6 5 1],3), clr, 'FaceAlpha',.3)
    patch(c([2 3 7 6 2],1), c([2 3 7 6 2],2), c([2 3 7 6 2],3), clr, 'FaceAlpha',.3)
    patch(c([3 4 8 7 3],1), c([3 4 8 7 3],2), c([3 4 8 7 3],3), clr, 'FaceAlpha',.3)
    patch(c([1 4 8 5 1],1), c([1 4 8 5 1],2), c([1 4 8 5 1],3), clr, 'FaceAlpha',.3)
else
    plot3(c([1 2 3 4 1],1), c([1 2 3 4 1],2), c([1 2 3 4 1],3),lt, 'Marker','.', 'Color',clr,'LineWidth',2)
    plot3(c([5 6 7 8 5],1), c([5 6 7 8 5],2), c([5 6 7 8 5],3),lt, 'Marker','.', 'Color',clr,'LineWidth',2)
    for ii = 0:3
        plot3(c([1 5]+ii,1), c([1 5]+ii,2), c([1 5]+ii,3), lt, 'Marker','.', 'Color',clr,'LineWidth',2)
    end
end

