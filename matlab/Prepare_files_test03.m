% Prepare input files for 2D example based on a crossection of the Tule
% river example.
y = 3.989e+6;
x = [2.714e+5 3.213e+5];
xx = linspace(x(1),x(2),100);

%% Bottom elevation
Bot = read_Scattered( '/home/giorgk/CODES/NPSAT/examples/Tule/Tule_bot_elev.npsat' );
F = scatteredInterpolant(Bot.p(:,1), Bot.p(:,2), Bot.v(:,1));
yy = F(xx, ones(1,100)*y);
% print file
name_file = '/home/giorgk/CODES/NPSAT/examples/box2D/Bot_test03.npsat';
fid = fopen(name_file,'w');
fprintf(fid, 'SCATTERED\n');
fprintf(fid, '%d %d\n', [length(xx), 1]);
fprintf(fid, '%.3f %.3f\n', [xx;yy]);
fclose(fid);
%% Top elevation
Top = read_Scattered( '/home/giorgk/CODES/NPSAT/examples/Tule/Tule_top_elev.npsat' );
F = scatteredInterpolant(Top.p(:,1), Top.p(:,2), Top.v(:,1));
xx = linspace(x(1),x(2),100);
yy = F(xx, ones(1,100)*y);
% print file
name_file = '/home/giorgk/CODES/NPSAT/examples/box2D/Top_test03.npsat';
fid = fopen(name_file,'w');
fprintf(fid, 'SCATTERED\n');
fprintf(fid, '%d %d\n', [length(xx), 1]);
fprintf(fid, '%.3f %.3f\n', [xx;yy]);
fclose(fid);
%% Dirichlet BC
fid = fopen('/home/giorgk/CODES/NPSAT/examples/box2D/Dirichlet_BC_box03.npsat','w');
fprintf(fid, '1\n');
fprintf(fid, 'TOP %.1f %.1f 55\n', [round(xx(19)) round(xx(20))]);
fclose(fid);
%% KX
KX = read_Scattered( '/home/giorgk/CODES/NPSAT/examples/Tule/Tule_KX.npsat' );
data = xx;
for ii = 1:size(KX.v,2)
    F = scatteredInterpolant(KX.p(:,1), KX.p(:,2), KX.v(:,ii));
    yy = F(xx, ones(1,100)*y);
    data = [data;yy];
end
fid = fopen('/home/giorgk/CODES/NPSAT/examples/box2D/KX_box03.npsat','w');
fprintf(fid, 'SCATTERED\n');
fprintf(fid, '%d %d\n', [size(data,2), size(data,1)]);
fprintf(fid, '%.2f %.2f %.2f %.2f %.2f %.2f\n',data);
fclose(fid);
%% KZ
KZ = read_Scattered( '/home/giorgk/CODES/NPSAT/examples/Tule/Tule_KZ.npsat' );
data = xx;
for ii = 1:size(KZ.v,2)
    F = scatteredInterpolant(KZ.p(:,1), KZ.p(:,2), KZ.v(:,ii));
    yy = F(xx, ones(1,100)*y);
    data = [data;yy];
end
fid = fopen('/home/giorgk/CODES/NPSAT/examples/box2D/KZ_box03.npsat','w');
fprintf(fid, 'SCATTERED\n');
fprintf(fid, '%d %d\n', [size(data,2), size(data,1)]);
fprintf(fid, '%.2f %.2f %.2f %.2f %.2f %.2f\n',data);
fclose(fid);
%% RCH
rch = read_Scattered( '/home/giorgk/CODES/NPSAT/examples/Tule/Tule_rch.npsat' );
F = scatteredInterpolant(rch.p(:,1), rch.p(:,2), rch.v(:,1));
yy = F(xx, ones(1,100)*y);
fid = fopen( '/home/giorgk/CODES/NPSAT/examples/box2D/RCH_test03.npsat','w');
fprintf(fid, 'SCATTERED\n');
fprintf(fid, '%d %d\n', [length(xx), 1]);
fprintf(fid, '%.7f %.7f\n', [xx;yy*100]);
fclose(fid);



