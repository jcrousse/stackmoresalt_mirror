from helpers.data_generator_functions import load_resize_bw_image

def apply_clustering(img_ids, img_dict, clustering_model):
    clusters = {}
    img_ids_list = []
    for image in img_ids:
        image_array = load_resize_bw_image(img_dict[image])
        if type(clustering_model) is dict:
            cluster = clustering_model[image]
        else:
            cluster = clustering_model(image_array, image)
        if cluster not in clusters:
            clusters[cluster] = len(clusters)
            img_ids_list.append([image])
        else:
            img_ids_list[clusters[cluster]].append(image)
    return img_ids_list