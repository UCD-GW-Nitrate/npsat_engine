function urf = simplifyURF(urf, tol)
    % urf [Nx2] [t Val]
    %https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
    np = size(urf,1);
    dmax = 0;
    index = 0;
    for ii = 2:np-1
        d = PerpendicularDistance(urf(ii,:), urf(1,:), urf(end,:));
        if d > dmax
            index = ii;
            dmax = d;
        end
    end
    
    if dmax > tol
        urf1 = simplifyURF(urf(1:index,:), tol);
        urf2 = simplifyURF(urf(index:end,:), tol);
        urf = [urf1; urf2];
    else
        urf = [urf(1,:); urf(end,:)];
    end
    
end

function dst = PerpendicularDistance(p, a, b)
    %https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    dst = abs((b(2) - a(2))*p(1) - (b(1) - a(1))*p(2) + b(1)*a(2) - b(2)*a(1))/sqrt((b(2) - a(2))^2 + (b(1) - a(1))^2);

end



