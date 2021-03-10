

function PCA(X)
    #get mean matrix of data
    mean_vec = mean(X, dims=2)
    #mormalize the data
    X = X .- mean_vec
    #compute covariance matrix
    M = X'*X
    #compute eigenvalues
    e, EV = eigen(M)
    #generate eigenvectors
    tmp = (X*EV)'
    V = tmp[end:-1:1,:]
    #generate vector of absolute value of eigenvalues
    S = reverse(sqrt.(abs.(e)))
    #order by size of eigenvalue?
    for i in 1:size(V, 2)
        V[:,i] ./= S
    end
    V, S, mean_vec
end

function get_projection(image_matrix, d)
    V, S, mean_vec = PCA(image_matrix)
    V'[:,1:d], mean_vec
end


function image_to_eigenface_rep(image, proj, mean_vec)
    proj'*(reshape(convert(Array{Float64}, image), :, 1) - mean_vec)
end


function eigenface_rep_to_image(eigenface_rep, proj, mean_vec, dims)
    image = proj*eigenface_rep + mean_vec
    Gray.(N0f8.(clamp01.(reshape(image, dims[1], dims[2]))))
end


function get_eigenfaces(proj, d, dims)
    eig(n) = rescale(proj[:,n])
    eigenfaces = reduce(hcat,[eig(i) for i in 1:d])
    image_matrix_to_images(eigenfaces, dims)
end
