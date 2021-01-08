
module MTHE493

using Images
using Statistics
using LinearAlgebra


include("eigenfaces.jl")
include("classification.jl")
include("utils.jl")


export  load_image,
        load_images,
        images_to_image_matrix,
        image_matrix_to_images,
        tile_images,
        rescale,
        get_projection,
        get_eigenfaces,
        image_to_eigenface_rep,
        eigenface_rep_to_image,
        euclidean_distance,
        k_nearest_neighbour,
        test_model


end
