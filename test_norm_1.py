from numpy import linalg as LA
import pandas as pd
import numpy as np
import random
import math
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import roc_auc_score
#########################################################################
######################use semi supervised learning######################
##input pairs of nodes output 0 or 1, whether 2 nodes belongs to the same person

#df =pd.read_table('node2vec/emb/test_edgelist.emb', sep = ' ', header = None)
#mat = np.zeros(shape=(70,70) )
df_bipart = pd.read_table('node2vec/emb/49070.emb', sep = ' ', header = None)
# get ground truth and sample some data
truth = pd.read_table('truth_49070.txt', sep = ',')
# generate a set of y  and X train data and label data. -1 means not labeled.
l = len(truth)
sample_size = int(math.floor(l/2))
random.seed(7)
sample_index = np.asarray(random.sample(range(l), sample_size))
X_id = []
y = []
labels = []
for i in range(l-1) :
    for j in range(i+1, l) :
        X_id.append([truth['ID'][i], truth['ID'][j]])
        if (truth['truth'][i] == truth['truth'][j]):
            y.append(1)
        else :
            y.append(0)
        if np.in1d(i, sample_index) and np.in1d(j, sample_index):
            labels.append(y[-1])
        else :
            labels.append(-1)

# generate X_train using X_id
#ID_list = df.ix[:,0]
ID_bipart = df_bipart.ix[:,0]
X_train = []
for i in range(len(X_id)):
    ID_1 = X_id[i][0]
    ID_2 = X_id[i][1]
    index_1 = np.where(ID_bipart == ID_1 )[0][0]
    index_2 = np.where(ID_bipart == ID_2 )[0][0]
    f1 = df_bipart.iloc[index_1][1:]
    f2 = df_bipart.iloc[index_2][1:]
    X_train.append(abs(f1-f2))


X_train= np.asarray(X_train)
# find the cut
cut = float(len(np.where(np.asarray(labels)==1 )[0]))/(len(np.where(np.asarray(labels)==0 )[0]) + len(np.where(np.asarray(labels)==1 )[0]))

########################build semi-supervied model #########################
label_spread = label_propagation.LabelSpreading(kernel = 'rbf', n_neighbors=7, alpha = 0.8)
label_spread.fit(X_train, labels)

y_fitted = label_spread.predict_proba(X_train)
y_fitted_s = []
label_spread.score(X_train, y)
fpe =0
fne =0

for i in range(len(y_fitted)):
    if y_fitted[i][1] > cut:
        y_fitted_s.append(1)
    else :
        y_fitted_s.append(0)


for i in range(len(y_fitted_s)):
    if y_fitted_s[i] -y[i] >0:
        fpe +=1 # 0 predict to 1
    if y_fitted_s[i] -y[i] <0:
        fne +=1 # 1 predict to 0



fpe 
fne
y_fitted
index_test= np.where(np.asarray(labels) == -1)[0]
roc_auc_score(y([index_test][0]),y_fitted_s[index_test])
sum(y_fitted_s)

if False:'''
label_prop_model = LabelSpreading(kernel = 'knn', n_neighbors = 4, alpha = 0.155)
label_prop_model.fit(X_train, labels)
y_fitted = label_prop_model.predict_proba(X_train)
'''




if False :'''
## norm of each pairs of vectors
for i in range (len(df)):
    for j in range(len(df)) :
        a = abs(df.iloc[i][1:128] - df.iloc[j][1:128])
        mat[i,j] = LA.norm(a)

# print pairwised data
for i in  range(70):
    for j in range(i+1,70):
        if abs (mat[i,j]) < 0.3:
            print (df.ix[i,0], df.ix[j,0])


sample = [[1066017246744202,'mostafa.mosad3'],[975611912520299,'mostafa.mosad.10'],[ 1036909843057172,'mostafa.mosad.10' ],[190866118021578,'mostafa.mosad.1804'],[ 1153797658036234,'mostafa.mosad.9889'],[ 1170003323079675,'mostafa.mosad.7' ]]

sample_vec = []
#print sample[5][0]
for i in range (len(sample)):
    sample_vec = sample_vec + ( df[df.ix[:,0]==sample[i][0]].index.tolist())

print sample_vec
for i in  range(70):
    min_dist = []
    for j in range(len(sample)):
        if (mat[i,sample_vec[j]]< 0.3):
            min_dist.append(sample[j][1])
            min_dist.append(mat[i,sample_vec[j]])
    print (df.ix[i,0],min_dist )        
'''









        
