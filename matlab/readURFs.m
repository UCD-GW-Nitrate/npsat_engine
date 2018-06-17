function WellURF = readURFs(filename)

topt.dx = 20; % [m]
topt.dt = 1; % [years]
topt.Ttime = 200; % [years]

fid = fopen(filename,'r');
cnter = 1;
prev_Eid = -9;
while ~feof(fid)
    C = fscanf(fid,'%d',3);
    E_id = C(1);
    Np = C(3);
    C = fscanf(fid,'%f',Np*4);
    C = reshape(C,4,Np)';
    urf = ComputeURF(C(:,1:3), C(:,4), topt);
    

    if prev_Eid ~= E_id
        if prev_Eid ~= -9
            display(prev_Eid);
            WellURF(cnter,1).id = prev_Eid;
            WellURF(cnter,1).URF = URF;
            cnter = cnter + 1;
        end
        prev_Eid = E_id;
        URF = urf;
    else
        URF = [URF; urf];
    end
        

end
fclose(fid);