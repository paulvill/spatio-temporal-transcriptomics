import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import time 
import ot 
import pandas as pd
import sys
import itertools
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


#### Code partially taken from Novosparc package available on Github

def tensor_square_loss_adjusted(C1, C2, T,p,q):


    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a**2) / 2

    def f2(b):
        return (b**2) / 2

    def h1(a):
        return a

    def h2(b):
        return b
    
    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC = constC1 + constC2
    
    tens = constC - np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()
    
    tens = constC - np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()
    
    return tens

def compute_cost(target, locations, num_neighbors_source = 10, num_neighbors_target = 10):


    # Shortest paths matrices at target and source spaces
    num_neighbors_target = num_neighbors_target # number of neighbors for nearest neighbors graph at target
    A_locations = kneighbors_graph(locations, num_neighbors_target, mode='connectivity', include_self=True)
    sp_locations = dijkstra(csgraph = csr_matrix(A_locations), directed = False,return_predecessors = False)

    sp_locations_max = np.nanmax(sp_locations[sp_locations != np.inf])
    sp_locations[sp_locations > sp_locations_max] = sp_locations_max #set threshold for shortest paths

    num_neighbors_source = num_neighbors_source # number of neighbors for nearest neighbors graph at source
    A_expression = kneighbors_graph(target, num_neighbors_source, mode='connectivity', include_self=True)

    sp_expression = dijkstra(csgraph = csr_matrix(A_expression), directed = False, return_predecessors = False) 
    sp_expression_max = np.nanmax(sp_expression[sp_expression != np.inf])
    sp_expression[sp_expression > sp_expression_max] = sp_expression_max #set threshold for shortest paths

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    cost_locations = sp_locations / sp_locations.max()
    cost_locations -= np.mean(cost_locations)
    cost_expression = sp_expression / sp_expression.max()
    cost_expression -= np.mean(cost_expression)

    return cost_expression, cost_locations

def create_space_distributions(num_locations, num_cells):
    
    """Creates uniform distributions at the target and source spaces.
    num_locations -- the number of locations at the target space
    num_cells     -- the number of single-cells in the data."""
    p_locations = ot.unif(num_locations)
    p_expression = ot.unif(num_cells)
    return p_locations, p_expression

def compute_lineage_matrix(name_t_prec,name_t):
    '''
    input : 
    - name_t (ns,1)
    - name_t_suiv (ns+k,1)
    
    Solve the problem between Y_t et Y_(t+1)
    '''
    M_lin = np.zeros((len(name_t_prec),len(name_t)))
    compteur = 0
    for i in range(len(name_t_prec)):

        ## Si le nom precedent est dans le nom au temps t
        #print("i",i)
        #print("i+compteur",i+compteur)
        if name_t[i+compteur].find(name_t_prec[i]) != -1:
            #On vérifie si meme taille ie si c'est le meme nom
            if len(name_t_prec[i])== len(name_t[i+compteur]):
                M_lin[i,i+compteur] = 1
            # Si une lettre en plus -> division cellulaire 
            elif ((len(name_t_prec[i]) +1) == len(name_t[i+compteur])) and (name_t[i+compteur][-1] !="x"):
                M_lin[i,i+compteur] = 1
                M_lin[i,i+compteur+1] = 1
                compteur += 1
            elif ((len(name_t_prec[i]) +1) == len(name_t[i+compteur])) or ((len(name_t_prec[i]) +2  == len(name_t[i+compteur])) and (name_t_prec[i].find("/") != -1)):
                M_lin[i,i+compteur] = 1
            elif  (name_t[i+compteur].find("/") != -1) and (name_t_prec[i].find("/") == -1):
                M_lin[i,i+compteur] = 1

        elif name_t[i+compteur].find(name_t_prec[i][:-2]) != -1 :
            if name_t[i+compteur][:-2] == name_t[i+compteur+1][:-2]:
                M_lin[i,i+compteur] = 1
                M_lin[i,i+compteur+1] = 1
                compteur += 1
            else :
                M_lin[i,i+compteur] = 1

    return M_lin


def distance_between_T(gamma,T,T_prec,name_t,name_t_prec)  :
     
    Mat_lin = compute_lineage_matrix(name_t_prec,name_t)
    T_prev = np.dot(Mat_lin.T,T_prec)

    ecart= abs(T -T_prev)

    ecart = ecart/ecart.sum(axis=1)[:,None] #contrainte sur les lignes

    ecart = gaussian_kernel(ecart,gamma)


    return ecart


def gaussian_kernel(matrix,gamma):
    matrix_scale = matrix/gamma

    m = np.exp(matrix_scale)

    return m
              
def distance_between_T_vitesse(gamma,T,T_prec,T_prec_prec,name_t,name_t_prec,name_t_prec_prec)  :
     
    Mat_lin_t = compute_lineage_matrix(name_t_prec,name_t)
    Mat_lin_t_prec = compute_lineage_matrix(name_t_prec_prec,name_t_prec)
    T_prev = np.dot(Mat_lin_t.T,T_prec)
    T_prev_prev = np.dot(Mat_lin_t_prec.T,T_prec_prec)

    ecart_T=T_prev - T_prev_prev
    T_chap = T_prev + ecart_T
    ecart = abs(T- T_chap)
    
    ecart = ecart/ecart.sum(axis=1)[:,None] #contrainte sur les lignes


    ecart = gaussian_kernel(ecart,gamma)

    return ecart

def distance_between_pos(T,T_prec,name_t,name_t_prec,X):
    
    Mat_lin = compute_lineage_matrix(name_t_prec,name_t)
    T_prev = np.dot(Mat_lin.T,T_prec)
    
    matrice_df_1 = pd.DataFrame(T, columns = [x for x in range(len(T[0]))])
    m_1 = np.zeros_like(matrice_df_1.values)
    m_1[np.arange(len(matrice_df_1)), matrice_df_1.values.argmax(1)] = 1
    pos_1 = np.dot(m_1,X)
    
    matrice_df_2 = pd.DataFrame(T_prev, columns = [x for x in range(len(T_prev[0]))])
    m_2 = np.zeros_like(matrice_df_2.values)
    m_2[np.arange(len(matrice_df_2)), matrice_df_2.values.argmax(1)] = 1
    pos_2 = np.dot(m_2,X)
    
    #ecart_M = np.linalg.norm(pos_1 - pos_2)
    ecart_M = ((pos_1-pos_2)**2)**0.5
    scaler = MinMaxScaler()
    #print(ecart_M)
    loss = np.dot(ecart_M,X.T)
    loss_norm = scaler.fit_transform(loss)
    return loss_norm

##### Construction of the successives "tensors"

def construct_tensor_T(all_Y,num_locations):
    liste_T = []
    for vecteur in all_Y :
        q,p = create_space_distributions(num_locations,len(vecteur))
        T = np.outer(p,q)
        liste_T.append(T)
    return liste_T

def construct_tensor_Y(Y):
    #Y = Y.drop("new_time",axis =1)
    liste_vecteurs =  [group[1] for group in Y.groupby(pd.Grouper(key='new_time'))]
    liste_Y = []
    for vecteur in liste_vecteurs :
        mean = vecteur.groupby(['lineage']).mean()
        one_Y = mean.values
        liste_Y.append(one_Y)
    return liste_Y

def construct_names(names_cells):
    liste_vecteurs =  [group[1] for group in names_cells.groupby(pd.Grouper(key='new_time'))]
    liste_name = []
    liste_number_name = []
    for vecteur in liste_vecteurs :
        name = vecteur['lineage'].tolist()
        name = sorted(list(set(name)))
        liste_name.append(name)
        liste_number_name.append(len(name))
    return liste_name, liste_number_name
    
def construct_target_grid(num_cells):
    """Constructs a rectangular grid. First a grid resolution is randomly
    chosen. grid_resolution equal to 1 implies equal number of cells and
    locations on the grid. The random parameter beta controls how rectangular
    the grid will be -- beta=1 constructs a square rectangle.
    num_cells -- the number of cells in the single-cell data."""

    grid_resolution = int(np.random.randint(1, 2+(num_cells/1000), 1))
    grid_resolution = 10
    #num_locations = len(range(0, num_cells, grid_resolution))

    #grid_dim = int(np.ceil((num_locations)*(1/3)))
    grid_dim = 100

    x = np.arange(0,grid_dim,grid_resolution)
    y = np.arange(0,grid_dim,grid_resolution)
    z = np.arange(0,grid_dim,grid_resolution)
    liste = [x,y,z]
    loc = list(itertools.product(*liste))
    #locations = np.array([(i, j, k) for i in x for j in y for k in z])
    locations = np.array(loc)
    return locations

