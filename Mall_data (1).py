
# coding: utf-8

# In[1]:

import pandas as pd
mall = pd.read_csv(r'E:\ML_Codes\mall-customers\Mall_Customers.csv')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt


# In[3]:

mall.head()


# In[4]:

mall.info()


# In[5]:

X = mall.loc[:,['Annual Income (k$)','Spending Score (1-100)']]


# In[6]:

X


# In[7]:

from sklearn.cluster import KMeans
loss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    loss.append(kmeans.inertia_)


# In[9]:

plt.plot(range(1,11),loss)
plt.show()


# In[11]:

kmeansmodel = KMeans(n_clusters = 5)
kmeansmodel.fit(X)
ykmeans = kmeansmodel.predict(X)


# In[13]:

ykmeans


# In[14]:

X.loc[0]


# In[15]:

#Earning high but spensing less
#Average in terms of earning and spending
#Earning high and spending high
#Earning less ans spensing high
#Earning less, spending less


# In[33]:

plt.scatter(X[ykmeans == 0].iloc[:,0] , X[ykmeans == 0].iloc[:,1], c= 'r',label='Cluster 1')
plt.scatter(X[ykmeans == 1].iloc[:,0] , X[ykmeans == 1].iloc[:,1], c= 'b', label='Cluster 2')
plt.scatter(X[ykmeans == 2].iloc[:,0] , X[ykmeans == 2].iloc[:,1], c= 'g', label='Cluster 3')
plt.scatter(X[ykmeans == 3].iloc[:,0] , X[ykmeans == 3].iloc[:,1], c= 'cyan', label='Cluster 4')
plt.scatter(X[ykmeans == 4].iloc[:,0] , X[ykmeans == 4].iloc[:,1], c= 'magenta', label='Cluster 5')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()


# In[42]:

#two coordinate
X
X[ykmeans == 0].iloc[:,1]


# In[43]:

data = pd.read_csv(r'E:\ML_Codes\dtc\regressive.csv')


# In[44]:

data.head()


# In[50]:

data.info()


# In[49]:

data['Time_taken'] = data['Time_taken'].fillna(data['Time_taken'].mean())


# In[54]:

data = pd.get_dummies(data, columns=['3D_available','Genre'],drop_first=True)


# In[57]:

#Features
X = data.drop(['Collection'],axis=1)
#Label
Y = data[['Collection']]


# In[60]:

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.15)


# In[96]:

from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)


# In[97]:

regtree.fit(xtrain,ytrain)


# In[98]:

ytrain_predict = regtree.predict(xtrain)
ytest_predict = regtree.predict(xtest)


# In[99]:

from sklearn.metrics import mean_absolute_error
mean_absolute_error(ytrain_predict,ytrain)


# In[100]:

mean_absolute_error(ytest_predict,ytest)


# In[107]:

#Plotting the tree created0
dot_data = tree.export_graphviz(regtree, out_file=None)
from IPython.display import Image
import pydotplus


# In[108]:

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# Controlling Tree Growth 

# In[109]:

#Maximum numbers of levels in a tree
regtree1 = tree.DecisionTreeRegressor(max_depth=3)
regtree1.fit(xtrain,ytrain)
dot_data = tree.export_graphviz(regtree1, out_file=None,
                                feature_names=xtrain.columns,filled=True)
graph1 = pydotplus.graph_from_dot_data(dot_data)
Image(graph1.create_png())


# In[110]:

#Minimum observation at internal node
regtree2 = tree.DecisionTreeRegressor(min_samples_split=24)
regtree2.fit(xtrain,ytrain)
dot_data = tree.export_graphviz(regtree2, out_file=None,
                                feature_names=xtrain.columns,filled=True)
graph2 = pydotplus.graph_from_dot_data(dot_data)
Image(graph2.create_png())


# In[111]:

#Minimum observation at leaf node
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf=23)
regtree3.fit(xtrain,ytrain)
dot_data = tree.export_graphviz(regtree3, out_file=None,
                                feature_names=xtrain.columns,filled=True)
graph3 = pydotplus.graph_from_dot_data(dot_data)
Image(graph3.create_png())


# In[112]:

#Computer vision library
import cv2


# In[113]:

#conda install -c conda-forge opencv


# In[115]:

loc = r'E:\ML_Codes\all1\train\cat.13.jpg'
f = cv2.imread(loc)


# In[116]:

f.shape


# In[117]:

plt.imshow(f)
plt.show()


# In[ ]:



