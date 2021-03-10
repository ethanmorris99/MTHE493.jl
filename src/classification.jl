

function euclidean_distance(vec1, vec2)
    sqrt(sum([(vec1[i]-vec2[i])^2 for i in 1:length(vec1)]))
end


function k_nearest_neighbour(eigenface_table, new_eigenface, k)
    n=length(eigenface_table)
    distances = [euclidean_distance(eigenface_table[i][2], new_eigenface) for i in 1:n]
    nearest_neighbours = [eigenface_table[i] for i in sortperm(distances)[1:k]]
    classes = unique([neighbour[1] for neighbour in nearest_neighbours])
    nearest = nearest_neighbours[1][1]
    inverse_distance(neighbour) = 1/(euclidean_distance(neighbour, new_eigenface))
    min = 0
    for class in classes
        freq = sum([inverse_distance(neighbour[2]) for neighbour in filter(x->x[1]==class, nearest_neighbours)])
        if freq > min
            min = freq
            nearest = class
        end
    end
    nearest
end


function test_model(proj, images, d, n, k)
    #n is number of training images of 10

    #get unique label for every subject
    labels = unique(map(x->x[2], images))
    #divide image data into trainging set
    training_set = reduce(vcat,[map(x->x[1], filter(x->x[2] == labels[i], images))[1:n+1] for i in 1:size(labels, 1)])
    #convert image data into matrix
    image_matrix = images_to_image_matrix(training_set)
    #
    transformed = transpose(transpose(proj[:,1:d])*image_matrix)
    eigenface_table = reduce(vcat, [[[labels[i+1] [transformed[j,:]]] for j in (i*(n+1)+1):(i*(n+1)+n+1)] for i in 0:size(labels, 1)-1])
    correct = 0
    for i in 1:length(training_set)
        #retrieve a eigenface to classify from test data
        test_eigenface = eigenface_table[i]
        data = reduce(vcat, [eigenface_table[1:i-1], eigenface_table[i+1:end]])
        #check if closest face is correct
        if k_nearest_neighbour(data, test_eigenface[2], k) == test_eigenface[1]
            #increment counter for correctly classified faces
            correct = correct + 1
        end
    end
    return correct/length(training_set)
end
