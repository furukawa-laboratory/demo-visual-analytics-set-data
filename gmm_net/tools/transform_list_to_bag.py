import numpy as np
def transform_list_to_bag(list_of_indexes,num_members):
    bag_of_members = np.empty((0, num_members))
    for indexes in list_of_indexes:
        one_hot_vectors = np.eye(num_members)[indexes]
        one_bag = one_hot_vectors.sum(axis=0)[None, :]
        bag_of_members = np.append(bag_of_members, one_bag, axis=0)
    return bag_of_members

