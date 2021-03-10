
using Pkg
Pkg.activate("./")
Pkg.instantiate()
using MTHE493
using Plots
using MultivariateStats
using ImageView
#%%


image_dir = "./data/small_dataset/"
dims = (112,92)
images = load_images(image_dir; dims=dims)

#number of training images out of ten

#splitting up image and labels
image_matrix = images_to_image_matrix(map(x->x[1], images))
labels = map(x->x[2], images)
label_map = Dict(String.(unique(labels)) .=> collect(1:length(unique(labels))))

#split trainging and testing data in half
training_set = image_matrix[:,1:2:end]
testing_set = image_matrix[:,2:2:end]



#Kernel PCA implementation

# train a kernel PCA model
M = fit(KernelPCA, image_matrix; maxoutdim=100)

# apply kernel PCA model to testing set
Yte = transform(M, testing_set)

# reconstruct testing observations (approximately)
Reconstructed_test = reconstruct(M, Yte)

origin = image_matrix_to_images(testing_set, dims)
tile_images(origin[1:5],5)

Recontructed_set = image_matrix_to_images(Reconstructed_test, dims)
Reconstruct = tile_images(Recontructed_set[1:5],5)


P = fit(PCA, image_matrix; maxoutdim=100)

Yt = transform(P, testing_set)

# reconstruct testing observations (approximatel
R_test = reconstruct(P, Yt)

origin_p = image_matrix_to_images(testing_set, dims)
tile_images(origin_p[1:5],5)

R_set = image_matrix_to_images(R_test, dims)
R = tile_images(R_set[1:5],5)
