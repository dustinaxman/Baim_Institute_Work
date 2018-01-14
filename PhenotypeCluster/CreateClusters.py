import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.cluster import SpectralClustering,AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#%%
def gower_distance(X):
    """
    This function expects a pandas dataframe as input
    The data frame is to contain the features along the columns. Based on these features a
    distance matrix will be returned which will contain the pairwise gower distance between the rows
    All variables of object type will be treated as nominal variables and the others will be treated as 
    numeric variables.
    Distance metrics used for:
    Nominal variables: Dice distance (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    Numeric variables: Manhattan distance normalized by the range of the variable (https://en.wikipedia.org/wiki/Taxicab_geometry)
    """
    individual_variable_distances = []

    for i in range(X.shape[1]):
        feature = X.iloc[:,[i]]
        if feature.dtypes[0] == np.object:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
        else:
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)

        individual_variable_distances.append(feature_dist)

    return np.array(individual_variable_distances).mean(0)

def plot_label_color(x1,x2,labelingLists):
    LABEL_COLOR_MAP = {0 : 'r',
                       1 : 'k',
                       2 : 'm',
                       3 : 'w',
                       4 : 'b',
                       5 : 'g',
                       6 : 'y',
                       7 : 'c',
                       }
    for labeling in labelingLists:
        label_color = [LABEL_COLOR_MAP[l] for l in labeling]
        plt.figure()
        plt.scatter(x1,x2 , c=label_color)

#%%
df=pd.read_csv("/Users/deaxman/Projects/DATA/General/titanic.csv")

df=df.drop(['name', 'cabin','ticket'], axis=1).dropna()  



#%%
dist_mat=gower_distance(df)
gamma=0.1
affinity_mat=np.exp(-gamma * dist_mat ** 2)


#%% Cluster data

spectral_clust=SpectralClustering(n_clusters=4,affinity='precomputed')
agglomerative_clust_1=AgglomerativeClustering(n_clusters=4,affinity='precomputed',linkage='complete')
agglomerative_clust_2=AgglomerativeClustering(n_clusters=4,affinity='precomputed',linkage='average')


spectral_clust.fit(affinity_mat)
agglomerative_clust_1.fit(affinity_mat)
agglomerative_clust_2.fit(affinity_mat)


#%% Dimension reduction
full_df=pd.get_dummies(df)
X=full_df.values
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


#%%
x1=principalDf['principal component 1'].values
x2=principalDf['principal component 2'].values
labelingLists=[spectral_clust.labels_,agglomerative_clust_1.labels_,agglomerative_clust_2.labels_]
plot_label_color(x1,x2,labelingLists)



