function create_big2DM(M)
    #creates 2D SRS matrix from M of size lat x lon x fraction x altitude x measurements x time
    
    m, n, f, a, p, t = size(M)
    big2DM = zeros(m * n * t, p)
    for j = 1 : p 
        big2DM[:, j] = vec(M[:, :, 1, 1, j, :])
    end
    for k = 1 : a
        for i = 1 : f
            if k+i > 2
                twoDM = zeros(m * n * t, p)
                for j = 1 : p
                    twoDM[:, j] = vec(M[:, :, i, k, j, :])
                end
                big2DM = cat(big2DM, twoDM, dims = 1)
            end
        end
    end

    return big2DM
end

function predict(x::Array, twoD_mat::Matrix)
    #prediction from an array
    out_x = vec(x)
    return twoD_mat'*out_x
end

function predict(x::CuArray, twoD_mat::CuArray)
    #prediction from a CuArray
    out_x = vec(x)
    return twoD_mat'*out_x
end