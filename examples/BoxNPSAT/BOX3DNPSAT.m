%% Aquifer outline
domain = [0 0; 5000 0; 5000 5000; 0 5000];
%% River segments
r1 = [1178 5000; 2427 3568; 1912 2877; 2244 1808; 3032 1200];
r2 = [5000 3442; 4052 2630; 4329 1463; 3200 1032];
%% Boundary conditions
bc_delta = 200;
bc_center = [3000 1000];
bc_poly = [...
           bc_center(1) -  bc_delta bc_center(2) -  bc_delta; ... 
           bc_center(1) +  bc_delta bc_center(2) -  bc_delta; ...
           bc_center(1) +  bc_delta bc_center(2) +  bc_delta; ...
           bc_center(1) -  bc_delta bc_center(2) +  bc_delta; ...
];
%% Recharge 
Rch = 0.00018; % m/day
Qrch = (5000*5000 - polyarea(bc_poly(:,1), bc_poly(:,2)))*Rch;
%% Stream recharge
% Let's suppose that there is an additional 20% of diffuse recharge coming
% from rivers
Qstream = Qrch*0.2;
r1_seg_len = sqrt(diff(r1(:,1)).^2 + diff(r1(:,2)).^2);
r2_seg_len = sqrt(diff(r2(:,1)).^2 + diff(r2(:,2)).^2);
% let's also assume that 70% of the total stream recharge comes from the
% river 1
Qstrm1 = Qstream * 0.7;
Qstrm2 = Qstream * 0.3;
%% Generate pumping wells
Tot_pump = Qrch + Qstream;
Tot_Qw = 0;
wells = [];
Rseg = [[r1(1:size(r1,1)-1,:) r1(2:size(r1,1),:)]; ...
        [r2(1:size(r2,1)-1,:) r2(2:size(r2,1),:)]];
while Tot_Qw < Tot_pump
    while true
        xw = 400 + (4600 - 400)*rand;
        yw = 400 + (4600 - 400)*rand;
        if ~isempty(wells) 
            mindst = min(sqrt((wells(:,1) - xw).^2 + (wells(:,2) - yw).^2));
            if mindst < 400
                continue;
            end
        end

        dst = sqrt((xw-bc_center(1)).^2 + (yw-bc_center(2)).^2);
        if dst < bc_delta+400
            continue;
        end
        dst = Dist_Point_LineSegment(xw,yw,Rseg);
        if min(dst) < 400
            continue;
        else
            break;
        end
        
    end
    qw = 100 + (500 - 100)*rand;
    slw = 10 + (100-10)*rand;
    r = (250 - slw)*rand;
    bw = -270 + r;
    tw = bw + slw;
    wells = [wells;xw yw tw bw -qw];
    Tot_Qw = Tot_Qw + qw;
end
wells(:,5) = wells(:,5).*(Tot_pump/sum(wells(:,5)));
%% plot 
clf 
hold on
plot(domain([1:4 1],1), domain([1:4 1],2), 'linewidth',2, 'DisplayName', 'No flow')
plot([r1(:,1);nan;r2(:,1)], [r1(:,2);nan;r2(:,2)], 'linewidth',2, 'DisplayName', 'Rivers') 
plot(bc_poly([1:4 1],1), bc_poly([1:4 1],2), 'linewidth',2, 'DisplayName', 'Constant head')
plot(wells(:,1), wells(:,2),'.', 'DisplayName', 'Wells')
legend('Location','northoutside','Orientation','horizontal')
axis equal
axis off
%% Write files
% wells
fid = fopen('box01_wells.npsat','w');
fprintf(fid, '%d\n', size(wells,1));
fprintf(fid, '%f %f %f %f -%f\n', wells');
fclose(fid);
%% Boundary Conditions
fid = fopen('box01_bc.npsat','w');
fprintf(fid, '1\n');
fprintf(fid, 'TOP %d %.2f\n', [size(bc_poly,1) 30]);
fprintf(fid, '%.2f %.2f\n', bc_poly');
fclose(fid);
%% Rivers
fid = fopen('box01_rivers.npsat', 'w');
fprintf(fid, '%d\n', length(r1_seg_len) + length(r2_seg_len));
perc = r1_seg_len/sum(r1_seg_len).*(0.3 + 0.4*rand(length(r1_seg_len),1));
perc = perc/sum(perc);
for ii = 1:length(r1_seg_len)
    fprintf(fid, '%d %.5f %.2f\n', [2 Qstrm1*perc(ii)/(50*r1_seg_len(ii)) 50]);
    fprintf(fid, '%.2f %.2f\n', r1(ii,:));
    fprintf(fid, '%.2f %.2f\n', r1(ii+1,:));
end
perc = r2_seg_len/sum(r2_seg_len).*(0.3 + 0.4*rand(length(r2_seg_len),1));
perc = perc/sum(perc);
for ii = 1:length(r2_seg_len)
    fprintf(fid, '%d %.5f %.2f\n', [2 Qstrm2*perc(ii)/(50*r2_seg_len(ii)) 50]);
    fprintf(fid, '%.2f %.2f\n', r2(ii,:));
    fprintf(fid, '%.2f %.2f\n', r2(ii+1,:));
end
fclose(fid);
%% Write initial particle locations
xx = 100:100:4900;
yy = 100:100:4900;
zz = -250:50:0;
[Xpart, Ypart, Zpart] = meshgrid (xx, yy, zz);
Nx = length(100:100:4900);
Ny = length(100:100:4900);
Nz = length(-250:50:0);
Xpart = reshape(Xpart,Nx*Ny*Nz,1);
Ypart = reshape(Ypart,Nx*Ny*Nz,1);
Zpart = reshape(Zpart,Nx*Ny*Nz,1);
Eid = nan(length(Xpart),1);
Sid = nan(length(Xpart),1);
% assign different eintity id for each level
for ii = 1:Nz
   idz = find(Zpart == zz(ii));
   Eid(idz) = ii;
   Sid(idz) = [1:length(idz)]';
end
%% write particle locations
fid = fopen('layer_part.npsat','w');
fprintf(fid, '%d\n', length(Eid));
fprintf(fid, '%d %d %.1f %.1f %.1f\n', [Eid Sid Xpart Ypart Zpart]');
fclose(fid);

    


