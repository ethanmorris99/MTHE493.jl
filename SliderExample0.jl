
using Pkg
Pkg.activate("./")
Pkg.instantiate()
using MTHE493
using Plots
using Interact



#%%
#Number of training data out of 10


image_dir = "./data/really_small_training/"
dims = (480,640)
images = load_images(image_dir; dims=dims)

#splitting up image and labels
image_matrix = images_to_image_matrix(map(x->x[1], images))
labels = map(x->x[2], images)
label_map = Dict(String.(unique(labels)) .=> collect(1:length(unique(labels))))
#training_set = reduce(vcat,[map(x->x[1], filter(x->x[2] == labels[i], images))[1:n+1] for i in 1:size(labels, 1)])
training_set = map(x->x[1],images)
N = 50
tile_images(training_set,10)

#initial projection
eig_vect, mean_face = get_projection(image_matrix, 10)


#slider to see charateristics of each eigenvector
@manipulate throttle=.05 for a1=-100:10:100, a2=-100:10:100, a3=-100:10:100, a4=-100:10:100, a5=-100:10:100
    slider_face = mean_face + a1*eig_vect[:,1] + a2*eig_vect[:,2] + a3*eig_vect[:,3] + a4*eig_vect[:,4] + a5*eig_vect[:,5]
    tile_images(image_matrix_to_images(slider_face, dims), 1)
end



#Sort based on the values of eigenvector number 1
e = 1

max_angle = 0
min_angle = 100

left_side = []
right_side = []
right_label = []
left_label = []


for i in 1:length(training_set)
    eig_face = image_to_eigenface_rep(training_set[i], eig_vect, mean_face)
    angle_value = eig_face[e]

    if angle_value >= 0
           push!(left_side, training_set[i])
           push!(left_label, parse(Float64, labels[i]))
           if angle_value >max_angle
               max_angle = angle_value
           end
    else
           push!(right_side, training_set[i])
           push!(right_label, parse(Float64, labels[i]))
           if angle_value < min_angle
               min_angle = angle_value
           end

    end

end


for i in 1:10
    print(issubset(i, right_label) )
    print(issubset(i, left_label) )
end

#PCA on subsets
L_eig_vect, L_mean_face = get_projection(images_to_image_matrix(left_side), 10)
R_eig_vect, R_mean_face = get_projection(images_to_image_matrix(right_side), 10)

test_image_dir = "./data/really_small_testing/"
test_images = load_images(test_image_dir; dims=dims)
d = 10
k=1

img = map(x->x[1],test_images)
test_label = map(x->x[2], test_images)

    #convert image data into matrix
image_matrix_left = images_to_image_matrix(left_side)
transformed_left = transpose(transpose(L_eig_vect[:,1:d])*image_matrix_left)


image_matrix_right = images_to_image_matrix(right_side)
transformed_right = transpose(transpose(R_eig_vect[:,1:d])*image_matrix_right)

    #testing



correct = 0
for i in 1:4
    test_eigface = image_to_eigenface_rep(img[i], eig_vect, mean_face)
    angle_value = test_eigface[1]
    if angle_value >= 0
        if nearest_neighbour(transformed_left, left_label, test_eigface) == parse(Float64, test_label[i])
            correct = correct + 1
        else
            print("Left:")
            print(test_label[i])
            print(" identified as ")
            print(nearest_neighbour(transformed_left, left_label, test_eigface))
            print("\n")
        end
    else
        if nearest_neighbour(transformed_right, right_label, test_eigface) == parse(Float64, test_label[i])
            correct = correct + 1
        else
            print("Right:")
            print(test_label[i])
            print(" identified as ")
            print(nearest_neighbour(transformed_left, left_label, test_eigface))
            print("\n")
        end
    end
end


correct





function nearest_neighbour(eigenface_table,label, new_eigenface)
    n=25
    min_label= label[1]
    min_dist = euclidean_distance(eigenface_table[1], new_eigenface)
    for i in 1:n
        distance = euclidean_distance(eigenface_table[i], new_eigenface)
        if min_dist>=distance
            min_dist=distance
            min_label = label[i]
        end
    end


    return min_label
end
