#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-danger">
# 
# <div style="font-size:22pt; line-height:25pt; font-weight:bold; text-align:center;">Common functions created by Willy LAO </div>
# 
# 
# 
# Last update: 17/05/2020
# 
# </div>

# In[2]:


################################################################

# Function call: name_var(variable, globals())
# Purpose      : Returns the name of a variable in a string

# Inputs       : variable  = function, variable ...
#              : globals() = leave it like this 
# Outputs      : Name of the variable in a string

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3

################################################################

def name_var(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


# In[3]:


############################################################################

# Function call: data_display(data)
# Purpose      : Display a table containing the different dataframes and their describe()

# Inputs       : data       = list of dataframes
#                data_names = list of the names of the dataframes 
# Outputs      : Display a table with the head() and describe() of the data

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3

############################################################################

def data_display_pandas_with_describe(data, data_names):    
    
    import pandas as pd
    import ipywidgets as widgets
    from ipywidgets import interactive
    
    dict_display = {}
    for j in range(len(data)) :
        data_j_head     = data[j].head()
        data_j_describe = data[j].describe()
        data_j          = pd.concat([data_j_head, data_j_describe], sort=False)  
        dict_display.update({data_names[j]: data_j})

    def f(Data):
        return dict_display[Data]
    
    widgets.interact(f, Data=data_names)
    return f         


# In[1]:


############################################################################

# Function call: data_display_pandas([df1, df2], ['df1_name', 'df2_name']
# Purpose      : Display a table containing the different dataframes

# Inputs       : data       = list of dataframes
#                data_names = list of the names of the dataframes 
# Outputs      : Display a table with the head() and describe() of the data

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3

############################################################################

def data_display_pandas(data, data_names):    
    
    import pandas as pd
    import ipywidgets as widgets
    from ipywidgets import interactive
    
    dict_display = {}
    for j in range(len(data)) :
        data_j_head     = data[j].head()
        dict_display.update({data_names[j]: data_j_head})

    def f(Data):
        return dict_display[Data]
    
    widgets.interact(f, Data=data_names)
    return f         


# In[5]:


############################################################################

# Function call: data_display(data)
# Purpose      : Returns the name of a variable in a string

# Inputs       : data       = list of DASK dataframes
#                data_names = list of the names of the DASK dataframes 
# Outputs      : Display a table with the head() and describe() of the data

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3

############################################################################

def data_display_dask(data, data_names):    
    
    from dask import dataframe as daskdf
    import ipywidgets as widgets
    from ipywidgets import interactive
    import pandas as pd
    
    dict_display = {}
    for j in range(len(data)) :
        data_j_head     = data[j].head()
        data_j_describe = data[j].describe().compute()
        data_j          = pd.concat([data_j_head, data_j_describe], sort=False)  
        dict_display.update({data_names[j]: data_j})

    def f(Data):
        return dict_display[Data]
    
    widgets.interact(f, Data=data_names)
    return f         


# In[7]:


#############################################################################################################
# Fonction qui appliquera une Analyse en Composantes Principales  (PCA)

# Inputs : n_components => nombre de composantes principales
#          df_scaled    => tableau de données normalisé

# Ouputs : pca         => contient toutes les informations de la PCA
#          table_pca   => tableau qui contient toutes les données projetées sur les composantes principales 
#          df_pca      => DataFrame qui contient toutes les données projetées sur les composantes principales 

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
##############################################################################################################


def apply_PCA(n_components, df_scaled):
    from sklearn.decomposition import PCA
    import pandas as pd
    
    pca = PCA(n_components = n_components)
    table_pca = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(table_pca, columns=df_scaled.columns[0:n_components])
    
    return (pca, table_pca, df_pca)


# In[8]:


#############################################################################################################
# Fonction qui calcul le taux de variance expliquée de chaque composante principale

# Inputs : pca : contient toutes les informations de la PCA

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
##############################################################################################################

def explained_variance_ratio(pca):
    for pc, variance_ratio in enumerate(pca.explained_variance_ratio_):
        print('The principal component n°%d represents a variance vatio of %.1f%%'%(pc, variance_ratio*100))


# In[9]:


#############################################################################################################
# Fonction qui affiche le taux cumulé de variance expliquée après une PCA

# Inputs : pca          => contient toutes les informations de la PCA
#          n_components => nombre de composantes principales

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
##############################################################################################################

def plot_cumulative_explained_variance(pca, n_components):    
    import matplotlib.pyplot as plt
    import numpy as np

    plt.bar(range(1,len(pca.explained_variance_ratio_ )+1),pca.explained_variance_ratio_)
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.xticks(np.arange(0,n_components,1))
    plt.yticks(np.arange(0,1,step=0.1))
    plt.plot(range(1,len(pca.explained_variance_ratio_ )+1),
          np.cumsum(pca.explained_variance_ratio_),
          c='red',
          label="Cumulative Explained Variance")
    plt.legend(loc='upper left')
    plt.title('Cumulative Explained variance')
    plt.show()


# In[10]:


#############################################################################################################
# Fonction qui affiche la distribution des composantes principales

# Inputs : table_pca => tableau qui contient toutes les données projetées sur les composantes principales 

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
##############################################################################################################

def distribution_PC(table_pca):
    import matplotlib.pyplot as plt
    
  ## distribution des composantes principales
    plt.boxplot(table_pca)
    plt.title('Distribution des composantes principales')
    plt.show()


# In[11]:


########################################################################################################################
# Fonction qui appliquera une Analyse en Composantes Principales  (PCA) et affiche le taux cumulé de variance 
# Ici, on choisit a posteriori le nombre de composantes principales que l'on veut garder, contrairement à apply_PCA

# Inputs : n_chosen_components => nombre de composantes principales
#          df_scaled           => tableau de données normalisé

# Ouputs : pca         => contient toutes les informations de la PCA
#          table_pca   => tableau qui contient toutes les données projetées sur les composantes principales 
#          df_pca      => DataFrame qui contient toutes les données projetées sur les composantes principales 

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
########################################################################################################################

def choose_n_components(n_chosen_components, df_scaled):

    pca, table_pca, df_pca = apply_PCA(n_chosen_components, df_scaled)

    print(('The cumulative variance ratio with %d principal components '
         'is %.2f %%'%(n_chosen_components, sum(pca.explained_variance_ratio_[0:n_chosen_components])*100)))

    return(pca, table_pca, df_pca)


# In[12]:


########################################################################################################################
# Fonction qui crée une heatmap pour savoir quelle est l'importance d'une feature pour une composante principale donnée

# Inputs : n_chosencomponents => nombre de composantes principales
#          df_scaled          => tableau de données normalisé
#          pca                => contient toutes les informations de la PCA
#          tag                => "Famille", "Société", "CITIZ" ou "whole dataset" si on prend toutes les données, 
#                                   pas seulement discriminées selon le "Tag"

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
########################################################################################################################

def feature_importance(df_scaled, pca, n_chosen_components, tag):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10,5)) 
    columns = df_scaled.columns 
    ax = sns.heatmap(pca.components_[:n_chosen_components],
                  cmap='coolwarm',
                  yticklabels=["PCA"+str(x) for x in range(1,n_chosen_components+1)],
                  xticklabels=[columns[i]+' ('+str(i+1)+')' for i in range(len(columns))],
                  linewidths=.5,
                  cbar_kws={"orientation": "vertical"})
    plt.yticks(rotation=0) 
    plt.title('Importance de chaque attribut, Tag : "' + tag + '"')
    ax.set_aspect("equal")


# In[13]:


########################################################################################################################
# Fonction qui crée un biplot (projection des individus sur les 2 premières composantes principales)

# Inputs : score  => Données projetées selon les 2 premières composantes principales
#          coef   => Données projetées selon les 2 premières composantes principales
#          method => 'PCA'
#          tag    => "Famille", "Société", "CITIZ" ou "whole dataset" si on prend toutes les données, 
#                       pas seulement discriminées selon le "Tag"

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
########################################################################################################################

def biplot(score,coeff, method, tag, labels=None):
    import matplotlib.pyplot as plt

    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.figure(figsize=(10,7))
    plt.scatter(xs * scalex,ys * scaley, c=labels)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5,
                 head_width=0.05, length_includes_head=True, head_length=0.05)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1), fontsize=18)
    plt.ylabel("PC{}".format(2), fontsize=18)
    plt.title('Biplot ' + method + ', Tag : "' + tag +'"', fontsize=18)
    plt.grid()


# In[14]:


########################################################################################################################
# Plot les individus selon les 2 premières composantes principales après une méthode de classification non supervisée 

# Inputs : table_pca   => tableau qui contient toutes les données projetées sur les composantes principales 
#          labels      => groupes assignés aux individus après une méthode de  classification non supervisée 
#          method      => 'PCA'
#          tag         => "Famille", "Société", "CITIZ" ou "whole dataset" si on prend toutes les données, 
#                            pas seulement discriminées selon le "Tag"

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
########################################################################################################################

def plot_2D(table_pca, labels, method, tag):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.scatter(table_pca[:,0], 
              table_pca[:,1], 
              c=labels, 
              cmap='rainbow')

    plt.title(method + ' clustering 2D, Tag : "' + tag + '"')
    plt.show()


# In[15]:


########################################################################################################################
# Plot les individus selon les 2 premières composantes principales après une méthode de classification non supervisée 

# Inputs : table_pca   => tableau qui contient toutes les données projetées sur les composantes principales 
#          labels      => groupes assignés aux individus après une méthode de  classification non supervisée 
#          method      => 'PCA'
#          tag         => "Famille", "Société", "CITIZ" ou "whole dataset" si on prend toutes les données, 
#                            pas seulement discriminées selon le "Tag"

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
########################################################################################################################


def plot_3D(table_pca, labels, method, tag):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("white")
    my_dpi=96
    plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)

    # Plot initialisation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(table_pca[:,0], table_pca[:,1],
            table_pca[:,2], c=labels, cmap="Set2_r", s=60)
  
    # make simple, bare axis lines through space:
    xAxisLine = ((min(table_pca[:,0]), max(table_pca[:,0])), (0, 0), (0,0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(table_pca[:,1]), max(table_pca[:,1])), (0,0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(table_pca[:,1]), max(table_pca[:,2])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
  
    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(method + ' clustering 3D, Tag : "' + tag + '"')


# In[16]:


########################################################################################################################
# Fonction qui appliquera une Classification Ascendante Hiérarchique et affiche le dendrogramme associé

# Inputs : table_pca   => tableau qui contient toutes les données projetées sur les composantes principales 

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
########################################################################################################################

def plot_ACH(table_pca):
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as shc

    plt.figure(figsize=(10, 7))
    plt.title("Customer Dendograms")
    dend = shc.dendrogram(shc.linkage(table_pca, method='ward'))


# In[17]:


########################################################################################################################
# Fonction qui retourne les groupes associés à chaque individu après l'ACH

# Inputs : n_clusters => nombre de clusters
#          table_pca  => tableau qui contient toutes les données projetées sur les composantes principales

# Outputs : labels_ACH => groupes associés à chaque individu après ACH

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
########################################################################################################################


def clusters_ACH(n_clusters, table_pca):
    from sklearn.cluster import AgglomerativeClustering

    cluster = AgglomerativeClustering(n_clusters=n_clusters, 
                                    affinity='euclidean', 
                                    linkage='ward')
    cluster.fit_predict(table_pca)
    labels_ACH = cluster.labels_
    
    return(labels_ACH)


# In[18]:


########################################################################################################################
# Fonction qui retourne les groupes associés à chaque individu après le DBSCAN

# Inputs : eps => hyperparamètre qui décide du nombre de clusters. Plus eps est proche de 0, plus il y a de clusters
#                 Plus esp est supérieur à 1, moins il y a de clusters
#          table_pca  => tableau qui contient toutes les données projetées sur les composantes principales

# Outputs : labels_DBSCAN => groupes associés à chaque individu après DBSCAN

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3.7.3
########################################################################################################################

def clusters_DBSCAN(eps, table_pca):
    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=eps, min_samples = 2)
    labels_DBSCAN = dbscan.fit_predict(table_pca)# plot the cluster assignments
    return(labels_DBSCAN)


# In[19]:


########################################################################################################################
# Fonction qui retourne les groupes associés à chaque individu après le KMeans
#
# Inputs : n_clusters => nombre de clusters
#          table_pca  => tableau qui contient toutes les données projetées sur les composantes principales
#
# Outputs : labels_KMeans => groupes associés à chaque individu après ACH
########################################################################################################################

def clusters_KMeans(n_clusters, table_pca):
    from sklearn.cluster  import  KMeans

    clust=KMeans(n_clusters=n_clusters)
    clust.fit(table_pca)
    labels_KMeans=clust.labels_
  
    return(labels_KMeans)

