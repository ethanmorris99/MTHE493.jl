using MTHE493
using Test

@testset "MTHE493.jl" begin
    # Small Dataset
    image_dir = "./data/small_dataset/"
    dims = (112,92)
    images = load_images(image_dir; dims=dims)
    @test length(images) == 400

    image_matrix = images_to_image_matrix(map(x->x[1], images))
    @test size(image_matrix) == (10304, 400)

    labels = map(x->x[2], images)
    @test typeof(labels) == Array{SubString{String},1}

    label_map = Dict(String.(unique(labels)) .=> collect(1:length(unique(labels))))
    @test label_map.count == 40

    proj, _ = get_projection(image_matrix, 400)
    @test size(proj) == (10304, 400)

    eigenfaces = get_eigenfaces(proj, 25, dims)
    @test size(eigenfaces[1]) == (112,92)

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
