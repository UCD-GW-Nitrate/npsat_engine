function WellURF = readURFs_ichnos(filename, opt)
% WellURF = readURFs(filename, opt)
% This function reads the *.urfs files which is the result of the gather 
% process of the NPSAT 
% For each streamline this function solves the Advection Dispersion Equation
% using the options defined in opt. 
% Using the options structure one can pass parameters to the ComputeURF 
% function that is called to simulate the 1D ADE 
% The paramteres that can pass are: 
%   alpha and beta parameters that control the longitudinal dispersivity
%   Lmin The streamline length that if a streamline has greater length than 
%   Lmin then the numerical solution is used. Otherwise the analytical 
%   solution is used to avoid numerical dispersion.

    topt.dx = 20; % [m]
    topt.dt = 1; % [years]
    topt.Ttime = 200; % [years]
    topt.Lmin = 200; %[m]
    topt.mult = 1000000;
    if ~isempty(opt)
        topt.aL.alpha = opt.alpha;
        topt.aL.beta = opt.beta;
        if isfield(opt, 'Lmin')
            topt.Lmin = opt.Lmin;
        end
        if isfield(opt, 'Ttime')
            topt.Ttime = opt.Ttime;
        end
        if isfield(opt, 'Agemin')
            topt.Agemin = opt.Agemin;
        end
        if isfield(opt, 'mult')
            topt.mult = opt.mult;
        end
    end

    cnt_urfs = 0;
    fid = fopen(filename,'r');
    if fid < 0
        WellURF = [];
        return;
    end
    cnter = 1;

    %clf
    %hold on
    % Set the true mode to true to compute the URFS
    true_mode = true;
    
    if true_mode
        WellURF = allocate_space([]);
    else
        WellURF = [];
    end
    while 1
        temp = fgetl(fid);
        if temp == -1
            break;
        end
        end_of_streamline = false;
        pp = nan(5000,3);
        vv = nan(5000,1);
        cnt = 1;
        while end_of_streamline == false
            C = textscan(temp,'%f',9);
            if C{1}(1) < 0
                C = textscan(temp,'%f %f %f %s',1);
                E_id = C{2};
                S_id = C{3};
                ex_r = parseExitReason(C{4});
                break;
            end
            pp(cnt,:) = C{1}(3:5);
            vv(cnt,:) = C{1}(6);
            cnt = cnt + 1;
            temp = fgetl(fid);
        end
        pp(cnt:end,:) = [];
        vv(cnt:end,:) = [];
        cnt_urfs = cnt_urfs + 1;
        fprintf('%d %d %d\n',[E_id S_id cnt_urfs])
        vv = vv./topt.mult;
        %if cnt_urfs > 200
        %    break;
        %end
        %plot3(C(:,1),C(:,2),C(:,3),'.')
        if true_mode
            id_rmv = sqrt(sum(pp(:,1:2).^2,2)) < 10;
            pp(id_rmv,:) = [];
            vv(id_rmv,:) = [];
            %Find those that are identical
            id_rmv = find(sqrt(sum(diff(pp).^2,2)) <0.001);
            pp(id_rmv,:) = [];
            vv(id_rmv,:) = [];
            %if sqrt(sum(C(:,1:3).^2)) < 0.1
            %    C(1,:) = [];
            %    warning('The first point of streamline is (0 0 0) and will be removed')
            %end
            vv  = sqrt(sum(vv.^2,2));
            
            WellURF(cnter,1).Eid = E_id;
            WellURF(cnter,1).Sid = S_id;
            WellURF(cnter,1).Exit = ex_r;
            if ex_r == 1 || ex_r == 2  || ex_r == 3
                WellURF(cnter,1).p_cds = pp(1,:);
                WellURF(cnter,1).v_cds = vv(1);
                WellURF(cnter,1).p_lnd = pp(end,:);
                WellURF(cnter,1).v_lnd = vv(end);
                L = cumsum(sqrt(sum((pp(2:end,:) - pp(1:end-1,:)).^2,2)));
                WellURF(cnter,1).L = L(end);
                WellURF(cnter,1).Age = sum(diff([0;L])./vv(1:end-1))/365;
                % WellURF(cnter,1).v_eff = v_eff(1);
                % WellURF(cnter,1).v_m = v_eff(2);
                topt.Ttime = min(1000,round(4*sum(diff([0;L])./vv(1:end-1))/365));
                topt.Ttime = ceil(topt.Ttime/100)*100;
                %display(topt.Ttime)
                urf = ComputeURF(pp, vv, topt);
                %plot(urf)
                %drawnow
                WellURF(cnter,1).URF = urf;
            else
                WellURF(cnter,1).p_cds = pp(1,:);
                WellURF(cnter,1).v_cds = vv(1);
                WellURF(cnter,1).p_lnd = nan;
                WellURF(cnter,1).v_lnd = nan;
                WellURF(cnter,1).L = nan;
                WellURF(cnter,1).Age = nan;
                WellURF(cnter,1).URF = [];%%%%%%urf;
            end
            cnter = cnter + 1;
            if cnter > size(WellURF,1)
                WellURF = allocate_space(WellURF);
            end
        else
            dst = sqrt(sum((pp(1:end-1,:)-pp(2:end,:)).^2, 2));
            cumsumdst = cumsum(dst);
            v = sum(vv(1:end-1) + vv(2:end),2)/2;
            if cumsumdst(end) < 50
                 WellURF(cnter,:) = [pp(end,:)  sum(dst./v)]; % cumsumdst(end)
                %plot3(C(:,1), C(:,2), C(:,3),'.-')
                %drawnow
               % stop = true;
                cnter = cnter + 1;
            end
            
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

function id = parseExitReason(er)
    if strcmp(er,'EXIT_TOP')
        id = 1;
    elseif strcmp(er, 'EXIT_SIDE')
        id = 2;
    elseif strcmp(er,'EXIT_BOTTOM')
        id = 3;
    elseif strcmp(er,'CHANGE_PROCESSOR')
        id = 0;
    elseif strcmp(er,'INIT_OUT')
        id = 4;
    elseif strcmp(er,'MAX_INNER_ITER')
        id = 7;
    elseif strcmp(er,'STUCK')
        id = 5;
    else
        id = 0;
       warning(['Uknown Exit Reason ', er]);
    end

end

