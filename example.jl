
using Pkg
Pkg.activate("./")
Pkg.instantiate()
using MTHE493
using Plots


#%%


image_dir = "./data/small_dataset/"
dims = (112,92)
images = load_images(image_dir; dims=dims)

image_matrix = images_to_image_matrix(map(x->x[1], images))
labels = map(x->x[2], images)
label_map = Dict(String.(unique(labels)) .=> collect(1:length(unique(labels))))
proj, mean_vec = get_projection(image_matrix, 400)
tile_images(get_eigenfaces(proj, 25, dims), 5)

img = images[1][1]
[img eigenface_rep_to_image(image_to_eigenface_rep(img, proj, mean_vec), proj, mean_vec, dims)]

gr()
performance = [test_model(proj, images, d, 3, 1) for d in 1:15]
plot(performance)
