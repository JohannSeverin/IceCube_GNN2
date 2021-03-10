import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.decomposition import PCA

from sklearn.neighbors import kneighbors_graph as knn


def NearestNeighbors(pos, k = 6):
    return knn(pos, k)
    

def PCA_gen(pos, k = 6, self_loop = False):
    """
    Generate adjacancy matrix by considering the principle component axis of the triggers in the events.
    Symmetric around the event, so there will be k // 2 neighbors before and after along the principle component.
    """

    # Use PCA to find principle component projection
    p_components = np.array(PCA(n_components = 1).fit_transform(pos)).flatten()

    a_idxs       = neighbors(p_components, self_loop, k)
    ones         = np.ones(shape = a_idxs.shape[0])

    a            = csr_matrix((ones, (a_idxs[:,0], a_idxs[:, 1])))

    return a



def neighbors(values, self_loop = False, k = 6):
    # Returns a N x 2 array, where N is the amount of connections
    # Pairs each node with up to k // 2 neighbors on both sides

    sorted_idxs  = np.argsort(values)

    N_nodes      = len(sorted_idxs)

    side_band    = k // 2

    sparse_idx   = []

    for i in range(-side_band, side_band + 1):
        if i == 0 and self_loop == False:
            continue
        else:
            idxs = np.arange(max(0, i), min(N_nodes, N_nodes - i))
            sparse_idx.append(np.vstack([sorted_idxs[idxs], np.roll(sorted_idxs, i)[idxs]]))

    sparse_idx = np.column_stack(sparse_idx).T
    return sparse_idx.astype(int)



