using MTHE493
using Test

@testset "MTHE493.jl" begin

    # Small Dataset
    image_dir = "./data/small_dataset/"
    dims = (112,92)
    images = load_images(image_dir; dims=dims)
    image_matrix = images_to_image_matrix(map(x->x[1], images))
    labels = map(x->x[2], images)
    label_map = Dict(String.(unique(labels)) .=> collect(1:length(unique(labels))))

    proj, _ = get_projection(image_matrix, 400)
    tile_images(get_eigenfaces(proj, 25, dims), 5)

end






#%%
#= Large Dataset
image_dir = "./data/large_dataset/"
dims = (196,180)
images = load_images(image_dir, 1:100; dims=dims)
image_matrix = images_to_image_matrix(map(x->x[1], images))
labels = map(x->x[2], images)
label_map = Dict(String.(unique(labels)) .=> collect(1:length(unique(labels))))
tile_images(image_matrix_to_images(image_matrix, dims), 10)
=#
