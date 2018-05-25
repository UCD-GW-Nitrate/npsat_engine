function OUT_TYPE = find_cell_types( S )

    OUT_TYPE = [];    
    for ii = 1:length(S)
        for jj = 1:length(S(ii,1).Cells)
            if ~isempty(S(ii,1).Cells(jj,1).Type)
                OUT_TYPE = [OUT_TYPE; ii jj S(ii,1).Cells(jj,1).Type];
            end
            
        end
    end
end

