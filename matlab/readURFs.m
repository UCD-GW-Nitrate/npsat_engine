function WellURF = readURFs(filename)

    topt.dx = 20; % [m]
    topt.dt = 1; % [years]
    topt.Ttime = 200; % [years]

    fid = fopen(filename,'r');
    cnter = 1;

    %clf
    %hold on

    WellURF = allocate_space([]);
    while 1
        temp = fgetl(fid);
        if temp == -1
            break;
        end
        C = textscan(temp,'%d',3);
        E_id = C{1}(1);
        S_id = C{1}(2);
        fprintf('%d %d\n',[E_id S_id])
        Np = C{1}(3);
        C = fscanf(fid,'%f',Np*4);
        temp = fgetl(fid);
        C = reshape(C,4,Np)';
        %plot3(C(:,1),C(:,2),C(:,3),'.')
        urf = ComputeURF(C(:,1:3), C(:,4), topt);

        WellURF(cnter,1).Eid = E_id;
        WellURF(cnter,1).Sid = S_id;
        WellURF(cnter,1).p_cds = C(1,1:3);
        WellURF(cnter,1).v_cds = C(1,4);
        WellURF(cnter,1).p_lnd = C(end,1:3);
        WellURF(cnter,1).v_lnd = C(end,4);
        WellURF(cnter,1).URF = urf;
        cnter = cnter + 1;
        if cnter > size(WellURF,1)
            WellURF = allocate_space(WellURF);
        end
    end
    fclose(fid);
    WellURF(cnter:end,:) = [];
end

function well = allocate_space(well)
    cnt = 10000;
    if isempty(well)
        
        well(cnt,1).Eid = [];
        well(cnt,1).Sid = [];
        well(cnt,1).p_cds = [];
        well(cnt,1).v_cds = [];
        well(cnt,1).p_lnd = [];
        well(cnt,1).v_lnd = [];
        well(cnt,1).URF = [];
    else
        Nsize = size(well,1);
        well(Nsize+cnt,1).Eid = [];
        well(Nsize+cnt,1).Sid = [];
        well(Nsize+cnt,1).p_cds = [];
        well(Nsize+cnt,1).v_cds = [];
        well(Nsize+cnt,1).p_lnd = [];
        well(Nsize+cnt,1).v_lnd = [];
        well(Nsize+cnt,1).URF = [];
    end
end