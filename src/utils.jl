

function load_image(
    filename::AbstractString;
    dims::Tuple{Int, Int}, grayscale=true
)
    image = grayscale ? Gray.(load(filename)) : load(filename)
    image[1:dims[1], 1:dims[2]]
end


function load_images(
    image_dir::AbstractString, v::UnitRange{Int};
    dims::Tuple{Int, Int}, grayscale=true
)
    files = readdir(image_dir, join=true, sort=false)[v]
    [
        (
            load_image(file; dims=dims, grayscale=grayscale),
            match(r"\/([a-zA-Z0-9_]+)\.", file).captures[1]
        )
        for file in files
    ]
end


function load_images(
    image_dir::AbstractString;
    dims::Tuple{Int, Int}, grayscale=true
)
    v = 1:length(readdir(image_dir, join=true, sort=false))
    load_images(image_dir, v; dims=dims, grayscale=grayscale)
end


function images_to_image_matrix(images)
    reduce(hcat, map(x->reshape(x, :, 1), convert.(Array{Float64}, images)))
end


function image_matrix_to_images(image_matrix, dims)
    images = [
        reshape(image_matrix[:, i], dims[1], dims[2])
        for i in 1:size(image_matrix, 2)
    ]
    [Gray.(image) for image in images]
end


function tile_images(images, cols)
    rows = size(images, 1) ÷ cols
    reduce(
        vcat,
        (reduce(hcat, (images[i*cols+j] for j = 1:cols)) for i = 0:rows-1)
    )
end


function rescale(data)
    min = minimum(data)
    max = maximum(data)
    scale(x) = (x-min)/(max-min)
    scale.(data)
end
