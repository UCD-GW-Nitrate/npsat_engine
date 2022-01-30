%% Generate wells
% 10ft of screen length for every 100 gpm
% 3.048 m                         545 m^3/day

Q_tot = 0;
wells = [];
while Q_tot < 5000
    while true
        xw = 400 + (4600 - 400)*rand;
        yw = 400 + (4600 - 400)*rand;
        if isempty(wells)
            break;
        else
            mindst = min(sqrt((wells(:,1) - xw).^2 + (wells(:,2) - yw).^2));
            if mindst > 400
                break;
            end
        end
    end
    
    qw = 100 + (500 - 100)*rand;
    slw = 10 + (100-10)*rand;
    r = (250 - slw)*rand;
    bw = -270 + r;
    tw = bw + slw;
    wells = [wells;xw yw tw bw -qw];
    Q_tot = Q_tot + qw;
end
wells(:,5) = wells(:,5).*(-5000/sum(wells(:,5)));
%% write wells to file
fid = fopen('wells_box3d.npsat','w');
fprintf(fid,'%d\n', size(wells,1));
fprintf(fid,'%f %f %f %f %f\n',wells');
fclose(fid);
%% Plot example
% domain
clf 
hold on
plot3([0 5000 5000 0 0],[0 0 5000 5000 0],[-270 -270 -270 -270 -270],'r')
plot3([0 5000 5000 0 0],[0 0 5000 5000 0],[30 30 30 30 30],'r')
plot3([0 0],[0 0],[-270 30],'r')
plot3([5000 5000],[0 0],[-270 30],'r')
plot3([5000 5000],[5000 5000],[-270 30],'r')
plot3([0 0],[5000 5000],[-270 30],'r')
% wells

for ii = 1:size(wells,1)
    plot3([wells(ii,1) wells(ii,1)], [wells(ii,2) wells(ii,2)],[wells(ii,3) wells(ii,4)],'o-')
end
%% define BC polygon on top
bc_poly = ginput(7);
[x2, y2] = poly2ccw(bc(:,1), bc(:,2));
plot3(bc_poly(:,1),bc_poly(:,2), 30*ones(7,1),'g')
%% list the coordinates
fprintf('%.2f %.2f\n', [x2 y2]')
%% one bc_poly example
bc_poly = [...
2506.15 2561.50;...
2709.10 2869.00;...
2819.80 2715.25;...
2616.85 1851.11;...
2034.24 2002.39;...
1930.73 1795.38;...
1731.69 1779.46];
%% left side boundary interpolation function
x = [-10:250:5250];
H(1,:) = 25*(0.3*([1:22].^2.1)/600+1);
EL(1,:) = -30*(0.7*([1:22]/22+1));
H(2,:) = 20*(0.3*([1:22].^2.1)/600+1);
EL(2,:) = -70*(0.7*([1:22]/22+1));
H(3,:) = 10*(0.3*([1:22].^2.1)/600+1);
EL(3,:) = -200*(0.5*([1:22]/22+1));
H(4,:) = 5*(0.3*([1:22].^2.1)/600+1);

bc_left = [cumsum([0 diff(x)])' H(1,:)' EL(1,:)' H(2,:)' EL(2,:)' H(3,:)' EL(3,:)' H(4,:)'];
%%
clf
hold on
EL(1,:) = -30*(0.7*([1:21]/21+1));
EL(2,:) = -70*(0.7*([1:21]/21+1));
EL(3,:) = -200*(0.5*([1:21]/21+1));
plot(x,EL(1,:),'b')
plot(x,EL(2,:),'g')
plot(x,EL(3,:),'r')
%% write left boundary to file
fid = fopen('box3d_leftv1.npsat','w');
fprintf(fid, 'SCATTERED\n');
fprintf(fid, 'VERT\n');
fprintf(fid, 'STRATIFIED\n');
fprintf(fid, '%d %d\n', [size(bc_left,1) 7]);
fprintf(fid, '%f %f %f %f %f %f %f %f\n',bc_left');
fclose(fid);
%% right side boundary
bc_right = [cumsum([0 diff(x)])' [40*(0.4*([1:22].^2.1)/600+1)]'];
fid = fopen('box3d_rightv1.npsat','w');
fprintf(fid, 'SCATTERED\n');
fprintf(fid, 'VERT\n');
fprintf(fid, 'SIMPLE\n');
fprintf(fid, '%d %d\n', [size(bc_right,1) 1]);
fprintf(fid, '%f %f\n',bc_right');
fclose(fid);
%% Test 4
% x y v
bc_pline = [...
2500 0 40;...
4000 0 30;...
5000 0 20;...
5000 1500 30;...
5000 2500 50 ...
];
%% write boundary line function file
fid = fopen('box3d_bnd_lines.npsat', 'w');
fprintf(fid, 'BOUNDARY_LINE\n');
fprintf(fid, '%d %d %f\n', [size(bc_pline,1) 1, 1.0]);
fprintf(fid, '%f %f %f\n', bc_pline');
fclose(fid);
%% Write files for Multipolygon recharge
% First split the domain
plot([0 5000 5000 0 0], [0 0 5000 5000 0]);

% make firts poly
[x, y] = ginput;
poly1 = [x y];
hold on
plot(poly1(:,1), poly1(:,2))

% set two more points to define the second polygon
[x, y] = ginput(2);
poly2 = [poly1(1:2,:); [x y]; [-100 -100]];
plot(poly2(:,1), poly2(:,2))
% make the other polygon as the intersection 
p1 = polyshape(poly1(:,1),poly1(:,2));
p2 = polyshape(poly2(:,1),poly2(:,2));
dom = polyshape([0 5000 5000 0 0], [0 0 5000 5000 0]);

p3 = subtract(dom,p1);
p3 = subtract(p3, p2);
%%
plot(p1)
hold on
plot(p2)
plot(p3)
plot([0 5000 5000 0 0], [0 0 5000 5000 0]);
%% Create interpolation functions for each subdomain
% The first will be scattered
% scatter points on the bounding box
xr = min(p1.Vertices(:,1)) + (max(p1.Vertices(:,1)) - min(p1.Vertices(:,1)))*rand(60,1);
yr = min(p1.Vertices(:,2)) + (max(p1.Vertices(:,2)) - min(p1.Vertices(:,2)))*rand(60,1);
R1 = 0.00001 + (0.0005 - 0.00001)*rand(60,1);
writeScatteredData('p1_rch.npsat', struct('PDIM', 2, 'TYPE', 'HOR', 'MODE', 'SIMPLE'), [xr yr R1]);
%% The second will be gridded with equal spacing and nearest interpolation
xg = linspace(min(p2.Vertices(:,1)), max(p2.Vertices(:,1)), 10);
yg = linspace(min(p2.Vertices(:,2)), max(p2.Vertices(:,2)), 7);
fid = fopen('p2_Xaxis.npsat','w');
fprintf(fid, 'CONST %d\n', length(xg));
fprintf(fid, '%.2f %.2f\n',[xg(1) diff(xg(1:2))]);
fclose(fid);

fid = fopen('p2_Yaxis.npsat','w');
fprintf(fid, 'CONST %d\n', length(yg));
fprintf(fid, '%.2f %.2f\n',[yg(1) diff(yg(1:2))]);
fclose(fid);

fid = fopen('p_Zaxis.npsat','w');
fprintf(fid, 'CONST %d\n', 1);
fprintf(fid, '%.2f %.2f\n', [0 10]);
fclose(fid);

frmt = '%.5f';
for ii = 1:(length(xg)-1)
    frmt = [frmt ' %.5f'];
end
frmt = [frmt '\n'];

p2_data = 0.00001 + (0.001 - 0.00001)*rand(10,7);

fid = fopen('p2_rch.npsat','w');
fprintf(fid,'NEAREST %d %d %d\n', [length(xg), length(yg) 1]);
fprintf(fid,'p2_Xaxis.npsat p2_Yaxis.npsat p_Zaxis.npsat\n');
fprintf(fid, frmt, p2_data');
fclose(fid);
%% The third will be a gridded with linear interpolation with variable step
xgv = min(p3.Vertices(:,1));
while xgv(end) < max(p3.Vertices(:,1))
    dx = 100 + (800 - 100)*rand;
    xgv = [xgv xgv(end) + dx];
end

ygv = min(p3.Vertices(:,2));
while ygv(end) < max(p3.Vertices(:,2))
    dy = 100 + (800 - 100)*rand;
    ygv = [ygv ygv(end) + dy];
end

fid = fopen('p3_Xaxis.npsat','w');
fprintf(fid, 'VAR %d\n', length(xgv));
fprintf(fid, '%.2f\n',xgv);
fclose(fid);

fid = fopen('p3_Yaxis.npsat','w');
fprintf(fid, 'VAR %d\n', length(ygv));
fprintf(fid, '%.2f\n',ygv);
fclose(fid);

pp = peaks(max(length(xgv), length(ygv)));
pp = (pp - min(pp,[],'all'))/(max(pp,[],'all') - min(pp,[],'all'))*0.001;

p3_data = pp(1:7,:)';

frmt = '%.5f';
for ii = 1:(length(xgv)-1)
    frmt = [frmt ' %.5f'];
end
frmt = [frmt '\n'];

fid = fopen('p3_rch.npsat','w');
fprintf(fid,'LINEAR %d %d %d\n', [length(xgv), length(ygv), 1]);
fprintf(fid,'p3_Xaxis.npsat p3_Yaxis.npsat p_Zaxis.npsat\n');
fprintf(fid, frmt, p3_data');
fclose(fid);
%% Print the main recharge file
fid = fopen('mult_var_rch.npsat','w');
fprintf(fid, 'MULTIPOLY\n');
fprintf(fid, '%d\n', 3); %Number of polygons
% poly1
fprintf(fid, '%d %s %s\n', size(p1.Vertices,1), 'SCATTERED', 'p1_rch.npsat');
fprintf(fid, '%.2f %.2f\n', p1.Vertices');
% poly2
fprintf(fid, '%d %s %s\n', size(p2.Vertices,1), 'GRIDDED', 'p2_rch.npsat');
fprintf(fid, '%.2f %.2f\n', p2.Vertices');
% poly3
fprintf(fid, '%d %s %s\n', size(p3.Vertices,1), 'GRIDDED', 'p3_rch.npsat');
fprintf(fid, '%.2f %.2f\n', p3.Vertices');
fclose(fid);
%% Read redist output files
[Vel, Graph] = readRedistVelGrph('output/box3d_testVelnew_0000.vel');
%% plot
plot3(Vel.XYZ(:,1), Vel.XYZ(:,2), Vel.XYZ(:,3),'.')
hold on
plot3(Graph.XYZ(:,1), Graph.XYZ(:,2), Graph.XYZ(:,3),'o')
%% Find the interpolation points for a given cell
icell = 22074;
vpnts = [];
% Add the velocities of the cell itself
for ii = 1:length(Graph.VellCell{icell,1})
    vpnts = [vpnts; Vel.XYZ(Graph.VellCell{icell,1}(ii) + 1,:)];
end
% add the velocities from the neighboring cells
for ii = 1:length(Graph.NeighCells{icell,1})
    ineigh = Graph.NeighCells{icell,1}(ii) + 1;
    for j = 1:length(Graph.VellCell{ineigh,1})
        vpnts = [vpnts; Vel.XYZ(Graph.VellCell{ineigh,1}(j) + 1,:)];
    end
end
clf
plot3(Graph.XYZ(icell,1), Graph.XYZ(icell,2), Graph.XYZ(icell,3),'o')
hold on
plot3(vpnts(:,1), vpnts(:,2), vpnts(:,3), '.')