
using Pkg
Pkg.activate("./")
Pkg.instantiate()
using MTHE493
using Plots
using Interact



#%%
#Number of training data out of 10
n = 7

image_dir = "./data/really_small_dataset/"
dims = (112,92)
images = load_images(image_dir; dims=dims)

#splitting up image and labels
image_matrix = images_to_image_matrix(map(x->x[1], images))
labels = map(x->x[2], images)
label_map = Dict(String.(unique(labels)) .=> collect(1:length(unique(labels))))
training_set = reduce(vcat,[map(x->x[1], filter(x->x[2] == labels[i], images))[1:n+1] for i in 1:size(labels, 1)])

#initial projection
eig_vect, mean_face = get_projection(image_matrix, 100)


#slider to see charateristics of each eigenvector
@manipulate throttle=.05 for a1=-10:1:10, a2=-10:1:10, a3=-10:1:10, a4=-10:1:10, a5=-10:1:10
    slider_face = mean_face + a1*eig_vect[:,10] + a2*eig_vect[:,12] + a3*eig_vect[:,13] + a4*eig_vect[:,14] + a5*eig_vect[:,15]
    tile_images(image_matrix_to_images(slider_face, dims), 1)
end



#Sort based on the values of eigenvector number 6
e = 6

max_angle = 0
min_angle = 100

left_side = []
right_side = []
right_label = []
left_label = []



for i in 1:100
    eig_face = image_to_eigenface_rep(training_set[i], eig_vect, mean_face)
    angle_value = eig_face[e]

    if angle_value >= 4
           push!(left_side, training_set[i])
           push!(left_label, labels[i])
           if angle_value >max_angle
               max_angle = angle_value
           end
       else
           push!(right_side, training_set[i])
           push!(right_label, labels[i])
           if angle_value < min_angle
               min_angle = angle_value
           end

       end

end


for i in 1:400
    number_as_string=split(labels[i],"s")
    number[i]=parse(Int, number_as_string)
end

issubset(, right_label)
issubset(1, left_label)
