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


