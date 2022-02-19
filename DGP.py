import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, vstack
from scipy.sparse.linalg import cg, spsolve_triangular, spsolve, splu
from scipy.linalg import cholesky
import scipy.io as sio
import cv2
import time
import argparse


def load_normal_mask(normal_path, mask_path):
    if normal_path[-4:] == '.npy':
        normal = np.load(normal_path)
    elif normal_path[-4:] == '.mat':
        normal = sio.loadmat(normal_path)
        normal = normal['normal']
    else:
        normal = cv2.cvtColor(cv2.imread(normal_path), cv2.COLOR_BGR2RGB)
    
    if mask_path[-4:] == '.npy':
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(mask_path)
    
    if mask.ndim == 2:
        pass
    elif mask.shape[-1] == 3:
        mask = np.linalg.norm(mask, axis=-1)
    else:
        mask = np.linalg.norm(mask, axis=0)
    mask = (mask != 0)

    return normal, mask

def write_obj(points, mask, depth, h_value, save_path):
    """
    write the 3d object into the obj. file
    Params:
    points -- index of the vertex shape=(r,c)
    mask -- mask of the object,should be boolen img
    depth -- depth map of the object
    h_value -- the actual width of per pixel, 1 is recommended
    save_path -- the saved file path of the obj
    The triangulation procedure is from https://github.com/gray0018/Discrete-normal-integration/blob/master/main.py
    """
    f = open(save_path, 'w')

    # vertexs
    for (r, c) in points:
        seq = 'v' + ' ' + str(float(r)*h_value) + ' ' + str(float(c)*h_value) + ' ' + str(depth[r, c]) + '\n'
        f.writelines(seq)

    # get vertexs index
    vidx = np.zeros_like(mask, dtype=np.uint32)
    vidx[mask] = np.arange(np.sum(mask)) + 1

    # cyclic shift to left by 1 pixel, the raw mask is right relative to right_mask
    right = np.roll(vidx, -1, axis=1)
    right[:, -1] = 0
    right_mask = right > 0

    # cyclic shift to up by 1 pixel, the raw mask is down relative to right_mask
    down = np.roll(vidx, -1, axis=0)
    down[-1, :] = 0
    down_mask = down > 0

    # first cyclic shift to the left and then right both by 1 pixel, the raw mask is right-down relative to the rd_mask
    rd = np.roll(vidx, -1, axis=1)
    rd = np.roll(rd, -1, axis=0)
    rd[-1, :] = 0
    rd[:, -1] = 0
    rd_mask = rd > 0

    up_tri = mask & rd_mask & right_mask 
    low_tri = mask & down_mask & rd_mask 

    # get a mesh constructed by a point along with its right point and right-down point
    mesh = np.vstack((vidx[up_tri], rd[up_tri], right[up_tri])).T
    for i, j, k in mesh:
        seq = 'f' + ' ' + str(i) + ' ' + str(j) + ' ' + str(k) + '\n'
        f.writelines(seq)
    # get a mesh constructed by a point along with its right point and right-down point
    mesh = np.vstack((vidx[low_tri], down[low_tri], rd[low_tri])).T
    for i, j, k in mesh:
        seq = 'f' + ' ' + str(i) + ' ' + str(j) + ' ' + str(k) + '\n'
        f.writelines(seq)

    f.close()


def DGP_solver(normal, mask, h_value=1, iters=5, eps=0.0871557, solver='Cholesky'):
    """
    reconstruct depth from normal
    use method in "Surface-from-Gradients: An Approach Based on Discrete Geometry Processing."
    Params:
    normal -- normal map with size(H, W, 3)
    mask -- ROI mask img of the object, size (H, W), boolen image
    h_value -- the parameter h in the paper, default is set to 1
    iter -- iteration time, default is set to 0.0871557, i.e. 5°; if eps is not, do not excute the outlier handle procedure
    solver -- method to solve the least square eqution, options:Cholesky, LU, direct, cg. But Cholesky is not implemented
    Rets:
    depth_map -- reconstructed depth map
    
    coordinate system:         facet:
        y|                      
         |                 v{i,j+1}_______v{i+1,j+1}
         |_______x                |       |
         /                        |       |
        /                  v{i,j} |_______|v{i+1,j}          
      z/ 

    Notice: the given normal map should be in the same coordinate system

    Thanks the code from "https://github.com/hoshino042/NormalIntegration/blob/main/methods/orthographic_DGP.py"
    for some inspiration which can be seen in this code, but the code above do not repuduce the paper correctly, so in this code I
    implement a more feasible one
    """
    # img size
    img_H, img_W = mask.shape

    # get facet index
    facet_idx = np.zeros_like(mask, dtype=np.int32)
    facet_idx[mask] = np.arange(np.sum(mask))

    # get the mask of four vertexs of all facets
    # top_left <-> v{i, j+1}, top_right <-> v{i+1, j+1}, bottom_left <-> v{i,j}, bottom_right <-> v{i+1,j}
    top_left_mask = np.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)
    top_right_mask = np.pad(mask, ((0, 1), (1, 0)), "constant", constant_values=0)
    bottom_left_mask = np.pad(mask, ((1, 0), (0, 1)), "constant", constant_values=0)
    bottom_right_mask = np.pad(mask, ((1, 0), (1, 0)), "constant", constant_values=0)

    # get the index of vertexs
    vertex_mask = np.logical_or.reduce((top_right_mask, top_left_mask, bottom_right_mask, bottom_left_mask))
    vertex_idx = np.zeros((img_H + 1, img_W + 1), dtype=np.int32)
    vertex_idx[vertex_mask] = np.arange(np.sum(vertex_mask))

    # get the amounts of facets and vertexs
    num_facet = np.sum(mask)
    num_vertex = np.sum(vertex_mask)

    # get the index of four vertexs for all facets
    top_left_vertex = vertex_idx[top_left_mask].flatten()
    top_right_vertex = vertex_idx[top_right_mask].flatten()
    bottom_left_vertex = vertex_idx[bottom_left_mask].flatten()
    bottom_right_vertex = vertex_idx[bottom_right_mask].flatten()
    # in facet_id_vertice_id, the row index is correspond to the facet index and the four elements in each row is four vertexs index belong to this facet
    facet_id_vertice_id = np.hstack((bottom_left_vertex[:, None],
                                        bottom_right_vertex[:, None],
                                        top_right_vertex[:, None],
                                        top_left_vertex[:, None]))

    # initialize the depth of the center of the facet c{i,j} to be zero
    depth = np.zeros(num_facet, dtype=normal.dtype)

    # get x,y,z components of the normal belongs to every facets 
    nx = normal[mask, 0]
    ny = normal[mask, 1]
    nz = normal[mask, 2]

    # detect the abnormal normals(outliers), substitute the outliers to the normal of current facet.Before the first iteration,
    # all facet normal is (0,0,1)
    if eps is not None:
        outlier_mask = nz < eps
        outlier_idx = np.where(outlier_mask)[0]
        nx[outlier_mask] = 0
        ny[outlier_mask] = 0
        nz[outlier_mask] = 1

    ############### local shaping -> global blending #############

    ############### Construct A and b of the linear system Ax=b ###############
    ### A is only constructed by the matrix N in the paper, which is a sparse matrix
    ### x is the depth of each vertexs after global blending
    ### b is a col vector construted by the relative vector Np{i,j}, p{i,j} is the projected facet after local shaping 
    ### m and n is the amounts of facets and vertexs, respectively
    ### shape: A(4mxn) x(nx1) b(4mx1)
    ###############################################
    ### We can construct A beforehand cause it is the same in every iteration, and b need to be updated during iteration
    ###############################################
    
    # In A and b construction, we construct them for 4 step(4 mxn matrix), the i mxn matrix is filled by product of the i row data of N and z(f_ij)
    # (the f_ij here is facet after global blending).This mxn matrix has four non-zero value(N's row elements), a row stand for a facet, the col index of the four values 
    # are correspond to the four vertex index of the facet. After product the i mxn matrix and x, we can get all the i component of the relative vectors for all facets.
    # The counterpart in b is the product of the i row data of N and p(f_ij)
    row_idx = np.arange(num_facet)
    row_idx = np.repeat(row_idx, 4)
    col_idx = facet_id_vertice_id.flatten()

    N1 = [0.75, -0.25, -0.25, -0.25] * num_facet
    A1 = coo_matrix((N1, (row_idx, col_idx)), shape=(num_facet, num_vertex))

    N2 = [-0.25, 0.75, -0.25, -0.25] * num_facet
    A2 = coo_matrix((N2, (row_idx, col_idx)), shape=(num_facet, num_vertex))

    N3 = [-0.25, -0.25, 0.75, -0.25] * num_facet
    A3 = coo_matrix((N3, (row_idx, col_idx)), shape=(num_facet, num_vertex))

    N4 = [-0.25, -0.25, -0.25, 0.75] * num_facet
    A4 = coo_matrix((N4, (row_idx, col_idx)), shape=(num_facet, num_vertex))

    A = vstack([A1, A2, A3, A4])

    if solver == 'Cholesky':
        # excute cholesky factorization to symmetric positive defined matrix A.T@A -> A=L@L.T
        # P.S scipy could not excute cholesky factorization for sparse matrix so we have to use other method
        raise NotImplementedError
        L = cholesky(A.T@A, lower=True) 
        L = csr_matrix(L)
    elif solver == 'LU':
        # A = csc_matrix(A)
        LU = splu(csc_matrix(A.T@A))

    for _ in range(iters):
        ############## Step 1. local shaping #############
        # get projected vertex according to equ.(2) in the paper. Here have two projection method, the latter one is from the paper 
        # and the former one is a substituable and effective.
        # projection_bottom_left = depth - (-0.5*nx*h_value - 0.5*ny*h_value) / nz
        # projection_bottom_right = depth - (0.5*nx*h_value - 0.5*ny*h_value) / nz
        # projection_top_right = depth - (0.5*nx*h_value + 0.5*ny*h_value) / nz
        # projection_top_left = depth - (-0.5*nx*h_value + 0.5*ny*h_value) / nz

        projection_bottom_left = depth
        projection_bottom_right = depth - nx*h_value / nz
        projection_top_right = depth - (nx*h_value + ny*h_value) / nz
        projection_top_left = depth - ny*h_value / nz

        ############# Step 2. global blending ############
        projection = np.array([
            projection_bottom_left, projection_bottom_right, projection_top_right, projection_top_left  
        ])

        b1 = (np.array([[0.75, -0.25, -0.25, -0.25]]) @ projection).T
        b2 = (np.array([[-0.25, 0.75, -0.25, -0.25]]) @ projection).T
        b3 = (np.array([[-0.25, -0.25, 0.75, -0.25]]) @ projection).T
        b4 = (np.array([[-0.25, -0.25, -0.25, 0.75]]) @ projection).T

        b = np.concatenate((b1, b2, b3, b4))

        if solver == 'Cholesky':
            # Ax=b -> L@L.T@x=b -> L@y=b -> L.T@x=y 
            raise NotImplementedError
            y = spsolve_triangular(L, b, lower=True)
            x = spsolve_triangular(L.T, y, lower=False)
        elif solver == 'cg':
            x, _ = cg(A.T @ A, A.T @ b, maxiter=1000, tol=1e-9)
        elif solver == 'direct':
            x = spsolve(A.T @ A, A.T @ b)
        elif solver == 'LU':
            x = LU.solve(A.T @ b)
            x = np.squeeze(x)

        # get the depth of c{i,j} by compute the mean value of four vertex of the facet
        depth_vertex_per_facet = x[facet_id_vertice_id]
        depth = np.mean(depth_vertex_per_facet, axis=-1)

        # Update the abnormal normal of the outlier facet
        # method: four vertex for a facet and three can determine a plane(i.e determine by two vector)
        # we use tow group and three points per group to get two normal and compute their mean value to get
        # the new normal of the outlier facet
        if eps is not None:
            outlier_facet_vertex_depth = depth_vertex_per_facet[outlier_mask, :]
            for i in range(outlier_facet_vertex_depth.shape[0]):
                v1 = np.array([h_value, 0, outlier_facet_vertex_depth[i, 0] - outlier_facet_vertex_depth[i, 1]])
                v2 = np.array([0, h_value, outlier_facet_vertex_depth[i, 2] - outlier_facet_vertex_depth[i, 1]]) 
                v3 = np.array([h_value, h_value, outlier_facet_vertex_depth[i, 3] - outlier_facet_vertex_depth[i, 1]]) 
                # [v1, v3]n = 0 for general solution n1, [v2, v3]n = 0 for general solution n2
                V = np.array([v1, v3])
                eval, evec = np.linalg.eig(V.T@V)
                n1 = evec[:, np.argmin(eval)]
                V = np.array([v2, v3])
                eval, evec = np.linalg.eig(V.T@V)
                n2 = evec[:, np.argmin(eval)]
                n = (n1 + n2) / 2
                # update
                nx[outlier_idx[i]], ny[outlier_idx[i]], nz[outlier_idx[i]] = n[0], n[1], n[2]

    depth_map = np.zeros(mask.shape)
    depth_map[mask] = depth

    return depth_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Surface-from-Gradients: An Approach Based on Discrete Geometry Processing.")
    parser.add_argument('--obj', default=None, help='the obj name in the data, the format should be objName_objNormalSuffix_objMaskSuffix')
    parser.add_argument('-n', '--normal_path', default='./data/bunny_normal.npy', help='the path of the normal map, npy,mat file and image is supported')
    parser.add_argument('-m', '--mask_path', default='./data/bunny_mask.png', help='the path of the object mask, npy file and image file is supported')
    parser.add_argument('-o', '--output_path', default='./data/bunny.obj', help='the path of the output obj file')
    parser.add_argument('-w', '--h_value', default=1, help='the size of width of one pixel')
    parser.add_argument('-i', '--iters', default=5, help='iteration time of the DGP algorithm')
    parser.add_argument('-s', '--solver', default='cg', help='the sparse linear system solver', choices=['cg', 'Cholesky', 'direct', 'LU'])
    par = parser.parse_args()

    if par.obj is not None:
        items = par.obj.split('_')
        normal_path = './data/' + items[0]  + '_normal' + '.' + items[1]
        mask_path = './data/' + items[0] + '_mask' + '.' + items[2]
        output_path = './data/' + items[0] + '.obj'
    else:
        normal_path = par.normal_path
        mask_path = par.mask_path
        output_path = par.output_path

    normal, mask = load_normal_mask(normal_path, mask_path)

    start_time = time.time()
    depth_map = DGP_solver(normal, mask, h_value=float(par.h_value), iters=int(par.iters), solver=par.solver)
    end_time = time.time()

    r, c = np.where(depth_map != 0)
    points = np.array([r, c]).T
    write_obj(points, mask, depth_map, float(par.h_value), output_path)

    print("Reconstruction Finish! Used Time:{:.5f}s".format(end_time-start_time))
