#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:24:29 2020

@author: juliepinol
"""
import novosparc as nc
import numpy as np
from ot.bregman import sinkhorn
import time 
import pandas as pd
import preprocessing_data as prep
from sklearn.metrics.pairwise import euclidean_distances
import sys 
import random as rd 
import math
import matplotlib
matplotlib.use("Qt5Agg")
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
#from torch import linalg as LA
import math

def function_loss(trans, cost_s, cost_t,mu_s,mu_t, trans_prec,trans_prec_prec, beta,X_bis,index_t):
    
    '''
    Obtention de la loss de la descente de gradient 
    
    Inputs :
        - trans : matrice T obtenue en sortie de sinkhorn
        - cost_s : matrice des couts - espace physique
        - cost_t : matrice des couts - espace transcriptomique
        - mu_s : distribution exemples espace physique
        - mu_t : distribution exemple espace transcriptomique
        - trans_prec : matrice T du temps précédent
        - trans_prec_prec : matrice T de t-2
        - beta : parametre controlant l'importance de la contrainte par rapport à D_1
        - X_bis : grille des positions
        - index_t : index des temps existants dans la séquence
    
    Outputs :
        - cost_all_sum() : valeur de la loss
        - d_1 : valeur de la première partie de la loss (pour comparaison avec sans contrainte)
        
 
    '''
    
    ##### Calcul de d_1 selon le lemme de Peyre (décomposition de la loss) ####
    ns = mu_s.size(0)
    nt = mu_t.size(0)
    
    f1 = (cost_s ** 2)/2
    f2 = (cost_t ** 2)/2
    
    
    constC1 = torch.mm(torch.mm(f1,mu_s.reshape(-1,1)),
                       torch.ones(nt).reshape(1,-1).double())
    
    constC2 = torch.mm(torch.ones(ns).reshape(-1,1).double(),
                       torch.mm(mu_t.reshape(1,-1),torch.t(f2).double()))
    
    tens = torch.mm(torch.mm(cost_s, trans),
                    torch.t(cost_t).double())
    
    #tens -= torch.min(tens)

    cost = constC1 + constC2 - tens
    #trans = torch.matmul(mu_s, torch.t(mu_t))
        
    pos_1 = torch.mm(trans,X_bis.double())
    
    #### Initialisation des premiers temps à partir des vrais positions
    if index_t == 2:
        
        pos_2 =  torch.tensor(liste_vecteur_pos_normaux[1])
        pos_3 = torch.tensor(liste_vecteur_pos_normaux[0])
    
    elif index_t == 3:
        pos_2 = torch.mm(trans_prec,X_bis.double())
        pos_3 =  torch.tensor(liste_vecteur_pos_normaux[1])


    else :
        pos_2 = torch.mm(trans_prec,X_bis.double())
        pos_3 = torch.mm(trans_prec_prec,X_bis.double())
        
    ### Calcul de la différence entre les "vraies pos" et les pos obtenues -> 2e contrainte
    dist = 0
    for i in range(len(pos_1)):
                 dist += abs(pos_1[i] - (2*pos_2[i]-pos_3[i]))
    dist = dist #/len(pos_1)
    cost_constraint = dist
    cost_all =  beta*dist.sum() + (1-beta)*cost.sum()
    d_1 = cost.sum()
    
    return cost_all.sum(),d_1


def Main_gromov_wasserstein_OT_before_2(cost_mat, Y,X, alpha_linear,beta_linear, loss_fun, epsilon,name_T, temps,
                                     max_iter=1000, tol=1e-9):
  
    '''
    Code repris et modifié du package Novosparc. Calcul de T grâce à Sinkhorn + descente de gradient
    
    Inputs :
        - cost_mat : matrice des coûts utilisés dans D_2
        - Y : ensemble des vecteurs transcriptomiques (1 vecteur = 1 ligne)
        - X : grille spaciale (1 ligne = 1 position)
        - alpha_linear : terme pondérant la contrainte de l'info à priori
        - beta_linear : terme pondérant la contrainte temporelle
        - loss_fun : square loss utilisée dans d_1 mais théoriquement autre chose est possible
        - epsilon : valeur de la régularisation entropique
        - name_T : liste contenant les noms des différentes cellules a la suite par temps
        - temps : liste des différents temps de la séquence
        - max_iter=1000 : itération maximale de sinkhorn 
        - tol=1e-9 : erreur utilisée pour stopper sinkhorn en avance
        
            
    '''
    
    compteur = 0
    for one_Y in Y :
        compteur += len(one_Y)
    num_cells = 331
    
    num_locations = X.shape[0]

    q, p = prep.create_space_distributions(num_locations, num_cells)

    cost_mat = np.asarray(cost_mat, dtype=np.float64)
      

    T = prep.construct_tensor_T(Y,num_locations)

    
    err = [1 for x in list(temps)] #les erreurs doivent toutes etre basses pour chaque temps -> toutes init à 1
    
    try:
            cost_mat_norm = cost_mat/ cost_mat.max()
    except:
            cost_mat_norm = cost_mat
            
    for t in list(temps):
        
                
        index_t = list(temps).index(t)
                        
        if index_t < 2 :
    
    
            cpt = 0
        
                                    
            C1,C2 = prep.compute_cost(target = Y[index_t], locations = X)
            q, p = prep.create_space_distributions(np.shape(C2)[0], np.shape(C1)[0])
            C1 = np.asarray(C1, dtype=np.float64)
            C2 =np.asarray(C2, dtype=np.float64)

                                    
            T_t = T[index_t]
            while err[index_t] > tol and cpt < max_iter:
                                        
        
                                        
                if loss_fun == 'square_loss':
                    tens = prep.tensor_square_loss_adjusted(C1, C2, T_t ,p,q)

                cost_mat_norm = np.ones(tens.shape)
            
                Tprev = T_t
            
                tens_all = (1-alpha_linear)*tens + alpha_linear*cost_mat_norm 

                T_t = sinkhorn(p, q, tens_all, epsilon)
              
                if cpt % 10 == 0:
                    err[index_t]= np.linalg.norm(T_t - Tprev)
                        
                cpt += 1

            T[index_t] = T_t
        
        
    cpt = 0
    
    #### boucle pour les temps à partir de 2
    # -> peut se synchroniser avec celle du dessus 
    for t in list(temps):
        index_t = list(temps).index(t)
        cpt = 0

        if index_t > 1 :
            T_t = T[index_t]
            #T_t = prep.distance_between_T_vitesse(100,T[index_t],T[index_t-1],T[index_t-2],name_T[index_t], name_T[index_t-1],name_T[index_t-2])  

            C1,C2 = prep.compute_cost(target = Y[index_t], locations = X)
            q, p = prep.create_space_distributions(np.shape(C2)[0], np.shape(C1)[0])
            C1 = np.asarray(C1, dtype=np.float64)
            C2 =np.asarray(C2, dtype=np.float64)
            
            while (err[index_t] > tol and cpt < max_iter):

                
                
                Tprev = T_t
                                
                if loss_fun == 'square_loss':
                    tens = prep.tensor_square_loss_adjusted(C1, C2, T_t ,p,q)

                cost_mat_norm = np.ones(tens.shape)


                tens_all = (1-alpha_linear)*tens + alpha_linear*cost_mat_norm 
                
                T_t = sinkhorn(p, q, tens_all, epsilon)
                
                d_1_avant =   tens.sum() 
                if cpt % 10 == 0:
                    print(cpt)
                    err[index_t] = np.linalg.norm(T_t - Tprev)
        
                cpt += 1
            
            #### Creation des tenseurs pour la descente de gradient 
            
            T_t_tensor = torch.tensor(T_t)
            T_t_tensor_func = T_t_tensor.requires_grad_()
            
            q = torch.tensor(q)
            p = torch.tensor(p)
            C1 = torch.tensor(C1)
            C2 = torch.tensor(C2)
            trans_prec = torch.tensor(T[index_t-1])
            trans_prec_prec = torch.tensor(T[index_t-2])
            X_bis = torch.tensor(X)    
            
            
            optimizer = torch.optim.Adam([T_t_tensor_func], lr=1e-4)
                
            ### Boucle pour la descente de gradient 
            
            for i in range(300):
                with torch.autograd.set_detect_anomaly(True):
                    nn.functional.softmax(T_t_tensor_func, dim = 1)
                    optimizer.zero_grad()
                    y,d_1 = function_loss(T_t_tensor_func, C1, C2,p,q, trans_prec,trans_prec_prec, beta,X_bis,index_t)
                    y.backward(retain_graph=True)
                    optimizer.step()
                    with torch.no_grad():
                        T_t_tensor_func.clamp_(0, 1)
            
                
            #### Repasser au format  numpy 
            
            T_t = T_t_tensor_func.detach().numpy()
            #T_t = T_t/num_cells
            #T_t = T_t/T_t.sum(axis=1)[:,None]*(1/num_cells) #contrainte sur les lignes
            #T_t = T_t/T_t.sum(axis=0)[None,:]*(1/num_locations)
                
            T[index_t] = T_t

        
    return T,X,err,d_1_avant,d_1

def gromov_wasserstein_adjusted_norm(cost_mat, C1, C2, alpha_linear,p, q, loss_fun, epsilon,
                                     max_iter=1000, tol=1e-9, verbose=False, log=False):

    '''
    OT utilisé dans Novosparc (repris du package)
    '''
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    cost_mat = np.asarray(cost_mat, dtype=np.float64)

    T = np.outer(p, q)  # Initialization


    cpt = 0
    err = 1
    
    try:
            cost_mat_norm = cost_mat/ cost_mat.max()
    except:
            cost_mat_norm = cost_mat
            
    if alpha_linear == 1:
        T = sinkhorn(p, q, cost_mat_norm, epsilon)
    else:        
        while (err > tol and cpt < max_iter):
            
            Tprev = T

            if loss_fun == 'square_loss':
                tens = prep.tensor_square_loss_adjusted(C1, C2, T,p,q)

            tens_all = (1-alpha_linear)*tens + alpha_linear*cost_mat_norm

            T = sinkhorn(p, q, tens_all, epsilon )#, method = 'sinkhorn_epsilon_scaling')
        
            if cpt % 10 == 0:
            # We can speed up the process by checking for the error only all
            # the 10th iterations
                err = np.linalg.norm(T - Tprev)
                if log:
                    log['err'].append(err)

                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format(
                                'It.', 'Err') + '\n' + '-' * 19)
                        print('{:5d}|{:8e}|'.format(cpt, err))

            cpt += 1

    if log:
        return T, log
    else:
        return T, tens.sum()
    
    

##### ----------Creation des vecteurs transcriptomiques --------------

#### Nom du fichier des positions WT-EMBO1
file_name1 = '/Volumes/BLOUPI/stage_data/Data/position/WT-EMB01.txt'

#### Fichier excel de packer aax1971_Packer_Tables_S1_to_S6_S9_S12_S13_S15_S16.xlsx
file_name = "/Volumes/BLOUPI/stage_data/Data/lignage/aax1971_Packer_Tables_S1_to_S6_S9_S12_S13_S15_S16.xlsx" # path to file + file name
sheet =  "Table_S6"

#### Creation de la liste des noms des cellules et des noms des lignages
df = pd.read_excel(io=file_name, sheet_name=sheet)
data = df.values
names_packer = data[24:data.shape[0],0]
correspondance_packer = data[24:data.shape[0],6]
names_packer_df = pd.DataFrame(names_packer,columns = ['names'])


cell_lineage = pd.read_csv(file_name1, delimiter="\t", header=0)
cell_lineage_bis = pd.merge(left= cell_lineage, right=names_packer_df, left_on='cell_name', right_on='names')
        
print("chargement des donnees ok")

## Sélection d'une période temporelle
cell_lineage_plateau = cell_lineage_bis[cell_lineage_bis['time'].isin(np.arange(160,170,1))]

cell_lineage_plateau.drop(['names'],axis =1)
liste_vecteurs =  [group[1] for group in cell_lineage_plateau.groupby(pd.Grouper(key='time'))]

####  Sous échantillonnage des cellules aux cellules présentes à TOUS les pas de temps considérés
liste_noms = []
vecteur = liste_vecteurs[0]
liste_noms = vecteur["cell_name"].tolist()


liste_name = []
liste_vecteur_partial = []
for vecteur in liste_vecteurs :
    liste = vecteur["cell_name"].tolist()

    intersect_name = (list(set(liste_noms)&set(liste)))
    vecteur_partial = vecteur[vecteur['cell_name'].isin(intersect_name)]
    liste_name.append(vecteur_partial['cell_name'].tolist())
    liste_vecteur_partial.append(vecteur_partial)

vecteur_bis = liste_vecteur_partial[-1]
liste_noms_bis = vecteur_bis["cell_name"].tolist()

liste_name_bis = []
liste_vecteur_partial_bis = []

for vecteur in liste_vecteur_partial :
    liste = vecteur["cell_name"].tolist()

    intersect_name = (list(set(liste_noms_bis)&set(liste)))
    vecteur_partial = vecteur[vecteur['cell_name'].isin(intersect_name)]
    liste_name_bis.append(vecteur_partial['cell_name'].tolist())
    liste_vecteur_partial_bis.append(vecteur_partial)

cell_plateau = pd.concat(liste_vecteur_partial_bis,sort = True)    
#cell_plateau = cell_plateau.drop(["cell_name","names"],axis =1)
    
liste_cell_plateau = [group[1] for group in cell_plateau.groupby(pd.Grouper(key='time'))]

#### Generation des "vecteurs transcriptomiques"

n_components = 19000 #nb de coordonnées finales
n_features= 3 #nb de coordonnées initiales
random_matrix = np.random.normal(loc=0.0,scale= (1.0 / np.sqrt(n_components)),size=(n_components, n_features))

from sklearn import random_projection

### Matrice de rotation utilisée pour générer une rotation dans les vecteurs 3D initiaux
def matrice_rotation():
    R = np.zeros((3,3))
    R[0,0]=1
    R[1,1]= math.cos(0.3)
    R[1,2] = -math.sin(0.3)
    R[2,1]= math.sin(0.3)
    R[2,2] = math.cos(0.3)
    return R


#### Matrices utilisées pour générées une translation dans les données
def matrice_translation(nb_cells,nb_blocks):
    R = np.zeros((nb_cells,3))
    value_ligne = int(nb_cells/nb_blocks)
    i = 1
    while i < nb_blocks:
        R[(i-1)*value_ligne:i*value_ligne,:]= (i)*5
        i +=1
    R[(nb_blocks-1)*value_ligne:,:]= nb_blocks*5

    return R

def matrice_translation_bis(nb_cells,nb_blocks,nb_locations):
    R = np.zeros((nb_cells,nb_locations))
    value_ligne = int(nb_cells/nb_blocks)
    i = 1
    while i < nb_blocks:
        R[(i-1)*value_ligne:i*value_ligne,:]= (i)*5e-6
        i +=1
    R[(nb_blocks-1)*value_ligne:,:]= nb_blocks*5e-6

    return R

#### Génération des vecteurs transformés géométriquement puis génération des vecteurs transcriptomiques
liste_vecteur_fit = []

liste_name = []
liste_vecteur_pos_normaux =[]
nb_translation = 5
vecteur = liste_cell_plateau[0]
name = vecteur['cell_name'].tolist()
vecteur = vecteur.drop(["time",'names','cell_name'],axis =1)
vecteur_bis = vecteur.values
matrice_trans = matrice_translation(len(vecteur),5)

for index in range(len(liste_cell_plateau)) :
    liste_name.append(name)
    #vecteur_bis = vecteur_bis + matrice_trans
    vecteur_bis = np.dot(matrice_rotation(),vecteur_bis.T)
    vecteur_bis = vecteur_bis.T    
    vecteur_fit = np.dot(vecteur_bis,random_matrix.T) #si warning c'est normal on augmente la dimenssion alors que le package a été créé pour la diminuer

    liste_vecteur_pos_normaux.append(vecteur_bis)
    liste_vecteur_fit.append(vecteur_fit)

Y = liste_vecteur_fit


##### ----------CODE PRINCIPAL -------------- ########


## Les temps sont les memes que ceux des vecteurs spaciaux
liste_time = np.arange(160,170,1)


num_cells = len(Y)
    
    
X_1 = prep.construct_target_grid(331)
    
num_locations = X_1.shape[0]


loss_function = "square_loss"
alpha = 0
beta = 0.99
cost_marker_genes = np.ones((num_cells, num_locations))
liste_pair = []
eps = 5e-3
    
    #### OT before
transport, X_loc,err,d_1_OT_avant,d1_OT = Main_gromov_wasserstein_OT_before_2(1, Y = Y, X = X_1,  alpha_linear = alpha,beta_linear = beta , loss_fun = loss_function, epsilon= eps , name_T = liste_name, temps = liste_time)
print(err)
print("OT ok")
sys.stdout.flush()   

########################
## PREPARATION NOVOSPARC
########################
liste_novosparc  = [] 
liste_before = []
liste_before_weighted = []
liste_novosparc_weighted = []
liste_X1 = []


for i in range(0,len(liste_time)):
    print(i/(len(liste_time)-1)*100)


    Y_2 = Y[i]
    num_cells_2 = len(Y_2)
    cost_expression_2, cost_locations_2 = nc.rc.setup_for_OT_reconstruction(Y_2,
                                                                    X_1,
                                                                    num_neighbors_source = 10,
                                                                    num_neighbors_target = 10)

    p_locations_2, p_expression_2 = nc.rc.create_space_distributions(num_locations, num_cells_2)

    
    

    #################################
    ### initialisation des parametres 
    ### Lancement des algos
    ##################################
    

    matrice_df = pd.DataFrame(transport[i], columns = [x for x in range(len(transport[i][0]))])
    m = np.zeros_like(matrice_df.values)
    m[np.arange(len(matrice_df)), matrice_df.values.argmax(1)] = 1
    matrice_cell_weighted = np.dot(transport[i],X_1)
    matrice_cell_weighted = matrice_cell_weighted
    pos_1 = np.dot(m,X_1)

    #liste_pair.append((pos_1,pos_2))
    liste_before.append(pos_1)
    liste_before_weighted.append(matrice_cell_weighted)
    
    ## Novosparc 
    gw_2,d_1_novo = gromov_wasserstein_adjusted_norm(1, cost_expression_2, cost_locations_2,alpha, p_expression_2, p_locations_2,'square_loss', epsilon= eps, verbose = False)
    matrice_df = pd.DataFrame(gw_2, columns = [x for x in range(len(gw_2[0]))])
    m = np.zeros_like(matrice_df.values)
    m[np.arange(len(matrice_df)), matrice_df.values.argmax(1)] = 1
    pos_2 = np.dot(m,X_1)
    matrice_cell_weighted_2 = np.dot(gw_2,X_1)

    liste_novosparc.append(pos_2)
    liste_novosparc_weighted.append(matrice_cell_weighted_2)
    
    liste_X1.append(X_1)
    i +=1 


print(d_1_OT_avant,d1_OT)
print(d_1_novo)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


#### PLOT TRAJ DE TOUTES LES CELLS


color = ['yellow','b','lawngreen','lightgreen','forestgreen','midnightblue','b','cornflowerblue','steelblue','c']
color_1 = ['purple','m','fuchsia','deeppink','crimson']
color_2 = ['midnightblue','b','cornflowerblue','steelblue','c']


#### Obtenir les données transfformées géométriquement "ground truth"
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#yellowgreen

for i in range(len(liste_vecteur_pos_normaux)):
    ax.scatter(liste_vecteur_pos_normaux[i][:, 0], liste_vecteur_pos_normaux[i][:, 1],liste_vecteur_pos_normaux[i][:, 2],color= color[i],label = "time_" + str(i)) 
    #ax.scatter(liste_before_weighted[i][:, 0]*num_cells_2, liste_before_weighted[i][:, 1]*num_cells_2,liste_before_weighted[i][:, 2]*num_cells_2,c= color[i]) 
    #ax.scatter(liste_novosparc_weighted[i][:, 0]*num_cells_2, liste_novosparc_weighted[i][:, 1]*num_cells_2,liste_novosparc_weighted[i][:, 2]*num_cells_2,c= color_1[i]) 
    ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

plt.show()


liste_theorical_pos = []

for i in range(2,len(liste_vecteur_pos_normaux)):
    theorie_pos = (2*liste_vecteur_pos_normaux[i-1]-liste_vecteur_pos_normaux[i-2])
    liste_theorical_pos.append(theorie_pos)


#### obtenir toutes les cellules avec les 2 positions initiales
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(liste_vecteur_pos_normaux[0][:, 0],liste_vecteur_pos_normaux[0][:, 1],liste_vecteur_pos_normaux[0][:, 2],label = "time_" + str(0),color = 'b')
#liste_vecteur_pos_normaux[index_t-2]
ax.scatter(liste_vecteur_pos_normaux[1][:, 0],liste_vecteur_pos_normaux[1][:, 1],liste_vecteur_pos_normaux[1][:, 2],label = "time_" + str(1),color = 'm')

#ax.scatter(liste_theorical_pos[0][:, 0],liste_theorical_pos[0][:, 1],liste_theorical_pos[0][:, 2], color = "c")
#ax.scatter(liste_theorical_pos[1][:, 0],liste_theorical_pos[1][:, 1],liste_theorical_pos[1][:, 2], color = "c")
#ax.scatter(liste_theorical_pos[2][:, 0],liste_theorical_pos[2][:, 1],liste_theorical_pos[2][:, 2], color = "c")


for i in range(2,len(liste_vecteur_pos_normaux)):
    #ax.scatter(liste_vecteur_pos_normaux[i][:, 0], liste_vecteur_pos_normaux[i][:, 1],liste_vecteur_pos_normaux[i][:, 2],color= color[i],label = "time_" + str(i)) 
    ax.scatter(liste_before_weighted[i][65:72, 0], liste_before_weighted[i][65:72, 1],liste_before_weighted[i][65:72, 2],c= color[i],label = "time_" + str(i)) 
    #ax.scatter(liste_novosparc_weighted[i][70:80, 0]*num_cells_2, liste_novosparc_weighted[i][70:80, 1]*num_cells_2,liste_novosparc_weighted[i][70:80, 2]*num_cells_2,c= color[i]) 
    ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.show()

#### TRAJECTOIRE CELLULAIRE OT - > trajectoire de quelques cellules (8 ici)
liste_cell = []
liste_traj_cell = []
for i in range(65,73): #choix des cellules (la 8 au hasard)
    liste_1 = []
    liste_1.append(liste_vecteur_pos_normaux[0][i])
    liste_1.append(liste_vecteur_pos_normaux[2][i])
    for j in range(2,len(liste_vecteur_pos_normaux)):
        liste_1.append(liste_before_weighted[j][i])
    liste_cell.append(liste_1)

for i in range(len(liste_cell)):
    
    array_1 = np.stack(liste_cell[i],axis=1)
    liste_traj_cell.append(array_1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#print(liste_traj_cell[0][0])
for i in range(len(liste_traj_cell)):
    ax.plot(liste_traj_cell[i][0],liste_traj_cell[i][1],liste_traj_cell[i][2],color = 'g')

for i in range(2,len(liste_vecteur_pos_normaux)):
    #ax.scatter(liste_vecteur_pos_normaux[i][:, 0], liste_vecteur_pos_normaux[i][:, 1],liste_vecteur_pos_normaux[i][:, 2],color= color[i]) 
    ax.scatter(liste_before_weighted[i][65:72, 0], liste_before_weighted[i][65:72, 1],liste_before_weighted[i][65:72, 2],c= color[i],label = "time_" + str(i)) 
    #ax.scatter(liste_novosparc_weighted[i][70:80, 0]*num_cells_2, liste_novosparc_weighted[i][70:80, 1]*num_cells_2,liste_novosparc_weighted[i][70:80, 2]*num_cells_2,c= color[i]) 
    #ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

#ax.set_axis_off()
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.show()

########### ------------------ CALCUL DES METRIQUES --------- ####

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    '''
    Determine l'angle entre deux vecteurs'
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#liste_novosparc_weighted = liste_novosparc
#liste_before_weighted = liste_before

from scipy import stats

liste_finale_before = []
liste_finale_novosparc = []
liste_finale_reelle = []
liste_compteur_before = []
liste_compteur_novo = []
liste_finale_before_corr = []
liste_finale_novosparc_corr = []
liste_finale_true_corr = []
liste_diff_bef = []
liste_diff_novo = []
liste_diff_reel = []

for i in range(len(liste_before_weighted[0])): #nombre de lignes
    trajectoire_before =[]
    trajectoire_novosparc = []
    ground_truth = []
    for j in range(len(liste_before_weighted)-1) : #nombre de points par trajectoires 
        trajectoire_before.append((liste_before_weighted[j][i],liste_before_weighted[j+1][i]))
        trajectoire_novosparc.append((liste_novosparc_weighted[j][i],liste_novosparc_weighted[j+1][i]))
        ground_truth.append((liste_vecteur_pos_normaux[j][i],liste_vecteur_pos_normaux[j+1][i]))
    
    #Calcul des angles pour la trajectoire de chaque cellule
    liste_angles_before = []
    liste_angles_novosparc = []
    liste_angles_reels =[] 
    compteur_before = 0
    compteur_novo = 0
    for k in range(len(trajectoire_before)-1): #parcours des positions de chaque cellule 
        pair_1 = list(trajectoire_before[k]) #2 premieres positions
        pair_2 = list(trajectoire_before[k+1]) #2e et 3e positions
        ab = pair_1[1] - pair_1[0]
        ba = pair_1[0] - pair_1[1]

        bc = pair_2[1] - pair_2[0]

        if np.all(ab ==0) and np.all(bc == 0) : ## on verifie que les positions ne sont pas confondues sinon ca bug
            compteur_before +=1
        else :
            angle = angle_between(ab,bc)
            liste_angles_before.append(angle) 
        
        pair_1 = list(trajectoire_novosparc[k])
        pair_2 = list(trajectoire_novosparc[k+1])
        ab = pair_1[1] - pair_1[0]
        ba = pair_1[0] - pair_1[1]
        bc = pair_2[1] - pair_2[0]

        if np.all(ab ==0) and np.all(bc == 0) :
            compteur_novo +=1   
        else :
            angle = angle_between(ab,bc)
            liste_angles_novosparc.append(angle) 
        
        
        pair_1 = list(ground_truth[k])
        pair_2 = list(ground_truth[k+1])
        ab = pair_1[1] - pair_1[0]
        ba = pair_1[0] - pair_1[1]
        bc = pair_2[1] - pair_2[0]

        try :
            angle = angle_between(ab,bc)
            liste_angles_reels.append(angle)
        except :
            continue 
        
    liste_compteur_before.append(compteur_before)
    liste_compteur_novo.append(compteur_novo)

    if liste_angles_before :
        traj_before_array = np.mean(np.array(liste_angles_before))
        a = liste_angles_before[:-1]
        b = liste_angles_before[1:]
        a = np.array(a)
        b = np.array(b)
        #corr_1 = stats.pearsonr(a,b)
        liste_finale_before.append(traj_before_array)
        #liste_finale_before_corr.append(corr_1)
        diff = 0
        for i in range(1,len(liste_angles_before)):
            diff += abs(liste_angles_before[i]-liste_angles_before[i-1])
        liste_diff_bef.append(diff/(len(liste_angles_before)-1))
    if liste_angles_novosparc:
        traj_novosparc_array = np.mean(np.array(liste_angles_novosparc))
        c = liste_angles_novosparc[:-1]
        d = liste_angles_novosparc[1:]
        c = np.array(c)
        d = np.array(d)
        #corr_2 = stats.pearsonr(c,d)
        liste_finale_novosparc.append(traj_novosparc_array)
        #liste_finale_novosparc_corr.append(corr_2)
        diff = 0

        for i in range(1,len(liste_angles_novosparc)):
            diff += abs(liste_angles_novosparc[i]-liste_angles_novosparc[i-1])
        liste_diff_novo.append(diff/(len(liste_angles_novosparc)-1))
    if liste_angles_reels:
        ground_truth_array = np.mean(np.array(liste_angles_reels))
        liste_finale_reelle.append(ground_truth_array)
        e = liste_angles_reels[:-1]
        f = liste_angles_reels[1:]
        e = np.array(e)
        f = np.array(f)
        diff = 0
        for i in range(1,len(liste_angles_reels)):
            diff += abs(liste_angles_reels[i]-liste_angles_reels[i-1])
        liste_diff_reel.append(diff/(len(liste_angles_reels)-1))
        #corr_3 = stats.pearsonr(e,f)
        #liste_finale_true_corr.append(corr_3)
        
liste_finale_before = [incom for incom in liste_finale_before if str(incom) != 'nan']
dist_bef_mean = np.mean(np.array(liste_finale_before))
var_bef = np.std(np.array(liste_finale_before))


liste_finale_novosparc =[incom for incom in liste_finale_novosparc if str(incom) != 'nan']
dist_novo_mean = np.mean(np.array(liste_finale_novosparc))
var_novo = np.std(np.array(liste_finale_novosparc))


dist_ground_mean = np.mean(np.array(liste_finale_reelle))
var_ground = np.std(np.array(liste_finale_reelle))


#liste_finale_reelle =[incom for incom in liste_finale_reelle if str(incom) != 'nan']
#dist_ground_mean = np.mean(np.array(liste_finale_reelle))
print("bef",len(liste_finale_before),dist_bef_mean,var_bef)
sys.stdout.flush()
print("novo",len(liste_finale_novosparc),dist_novo_mean,var_novo)
sys.stdout.flush()


print("true",dist_ground_mean,var_ground)

print("diff angle bef",np.mean(np.array(liste_diff_bef)),np.std(np.array(liste_diff_bef)))
print("diff angle novo",np.mean(np.array(liste_diff_novo)),np.std(np.array(liste_diff_novo)))
print("diff angle reel",np.mean(np.array(liste_diff_reel)),np.std(np.array(liste_diff_reel)))
#print(np.mean(np.array(liste_finale_before_corr)))
#print(np.mean(np.array(liste_finale_novosparc_corr)))
#print(np.mean(np.array(liste_finale_true_corr)))

#### CALCUL DE LA DISTANCE ENTRE POS ET POS SI MOUV PREDIT
liste_dist_before = []
liste_dist_novo = []
## positions "ideales"
pos_chap_novo = liste_novosparc_weighted[1] + liste_novosparc_weighted[1] - liste_novosparc_weighted[0]
pos_chap_bef = liste_before_weighted[1]+ liste_before_weighted[1] - liste_before_weighted[0]
for i in range(2,len(liste_before_weighted)):
    dist_before = np.linalg.norm(liste_before_weighted[i]-pos_chap_bef)/np.linalg.norm(pos_chap_bef)
    pos_chap_bef = liste_before_weighted[i] + liste_before_weighted[i] -liste_before_weighted[i-1]
    
    dist_novo = np.linalg.norm(liste_novosparc_weighted[i]-pos_chap_novo)/np.linalg.norm(pos_chap_novo)
    pos_chap_novo = liste_novosparc_weighted[i] + liste_novosparc_weighted[i] -liste_novosparc_weighted[i-1]
    
    liste_dist_before.append(dist_before)
    liste_dist_novo.append(dist_novo)

mean_bef = np.mean(np.array(liste_dist_before))
std_bef = np.std(np.array(liste_dist_before))

mean_novo = np.mean(np.array(liste_dist_novo))
std_novo = np.std(np.array(liste_dist_novo))

print("dist theorique bef",mean_bef,std_bef)
print("dist theorique novo",mean_novo,std_novo)

