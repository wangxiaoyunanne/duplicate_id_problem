import pandas as pd
import igraph
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from numpy.linalg import pinv
from numpy import linalg as LA
from scipy import stats
from sklearn.preprocessing  import normalize

bipart = pd.read_table('sample_file/name_id_49070.txt', sep = '\t', header = None)
G = nx.from_pandas_dataframe(bipart, 0, 1)
A = nx.adjacency_matrix(G).todense()
# here A is symmetric matrix
def katz (MAT, norm = True):
    alpha = 1/(LA.norm(MAT))*0.99
    I = np.identity(MAT.shape[0])
    distence = pinv(I- alpha*MAT)- I
    if norm == False:
        return distence
    if norm == True:
        distence_norm = normalize(distence, norm = 'l1')
        return distence_norm

# row is normalized
K_A = katz(A)


############################################################################
p =1
q = 0.5
# using undirected unweighted graph
# generate transition matrix based on  node2vec
def alpha(G,p =1, q = 0.5):
    A = nx.adjacency_matrix(G).todense()
    result = np.zeros(A.shape)
    list_node = G.nodes()
    for t,v in G.edges_iter() :
        v_ind = list_node.index(v)
        t_ind = list_node.index(t)
########from t to v, current node is v and last step is in t
        v_neighbors = G.neighbors(v)
        t_neighbors = G.neighbors(t)
        Pv = 1.0/len(v_neighbors)
        Pt = 1.0/len(t_neighbors)
        #print (Pv,Pt)
        result[v_ind,t_ind] += 1.0/p* Pv
       # result[t_ind,v_ind] = 1/p* Pt2v
        x_1 = [node for node in v_neighbors if node in t_neighbors]
        if len(x_1):
            for node in x_1:
                node_ind = list_node.index(node)
                result[v_ind,node_ind] += 1.0* Pv
              #  result[node_ind,v_ind] = 1* Pt2v
              # here is from v to t, current node is t list is v
                result[t_ind,node_ind] += 1.0* Pt
              #  result[node_ind,t_ind] = 1* Pv2t
        x_2 = [node for node in v_neighbors if (node not in t_neighbors and node != t)]
        if len(x_2):
            for node in x_2:
                node_ind = list_node.index(node)
                result[v_ind,node_ind] += 1.0/q* Pv
               # result[node_ind,v_ind] = 1/q* Pt2v
########## then swith v and t, this time from v to t, current node is t 
        #result[v_ind,t_ind] = 1/p * Pv2t
        result[t_ind,v_ind] += 1.0/p * Pt
        # x_1 the same for v2t and t2v, X_22 here is X2 from t to v, current node is t
        x_22 = [node for node in t_neighbors if (node not in v_neighbors and node != v)]
        if len(x_22):
          for node in x_2:
                node_ind = list_node.index(node)
                result[v_ind,node_ind] += 1.0/q* Pv
               # result[node_ind,v_ind] = 1/q* Pt2v
########## then swith v and t, this time from v to t, current node is t 
        #result[v_ind,t_ind] = 1/p * Pv2t
        result[t_ind,v_ind] += 1.0/p * Pt
        # x_1 the same for v2t and t2v, X_22 here is X2 from t to v, current node is t
        x_22 = [node for node in t_neighbors if (node not in v_neighbors and node != v)]
        if len(x_22):
            for node in x_22:
                node_ind = list_node.index(node)
                result[t_ind,node_ind] += 1.0/q * Pt
               # result[node_ind,t_ind] = 1/q * Pv2t
    return result

data = pd.read_table('node2vec/graph/karate.edgelist', sep = ' ', header = None)
karate = nx.from_pandas_dataframe(data,0,1)
A = nx.adjacency_matrix(karate).todense()
result = np.zeros(A.shape)
K1 = alpha(karate)
#######################################################################
############get 
U, s, V = LA.svd(K1,full_matrices = False)




