function plot_min_max_cell( p,  q, clr )
%plot_min_max_cell plots a cell using the min p and max p points
if isempty(clr)
    clr = 'k';
end

hold on
plot3([p(1) p(1) q(1) q(1) p(1)], [p(2) q(2) q(2) p(2) p(2)], [q(3) q(3) q(3) q(3) q(3)], clr)
plot3([p(1) p(1) q(1) q(1) p(1)], [p(2) q(2) q(2) p(2) p(2)], [p(3) p(3) p(3) p(3) p(3)], clr)
plot3([p(1) p(1)], [p(2) p(2)], [p(3) q(3)], clr)
plot3([p(1) p(1)], [q(2) q(2)], [p(3) q(3)], clr)
plot3([q(1) q(1)], [p(2) p(2)], [p(3) q(3)], clr)
plot3([q(1) q(1)], [q(2) q(2)], [p(3) q(3)], clr)
hold off

