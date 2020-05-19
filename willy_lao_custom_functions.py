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

# In[ ]:


################################################################

# Function call: name_var(variable, globals())
# Purpose      : Returns the name of a variable in a string

# Inputs       : variable  = function, variable ...
#              : globals() = leave it like this 
# Outputs      : Name of the variable in a string

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3

################################################################

def name_var(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


# In[1]:


############################################################################

# Function call: data_display(data)
# Purpose      : Returns the name of a variable in a string

# Inputs       : data       = list of dataframes
#                data_names = list of the names of the dataframes 
# Outputs      : Display a table with the head() and describe() of the data

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3

############################################################################

def data_display_pandas(data, data_names):    
    
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


# In[ ]:


############################################################################

# Function call: data_display(data)
# Purpose      : Returns the name of a variable in a string

# Inputs       : data       = list of DASK dataframes
#                data_names = list of the names of the DASK dataframes 
# Outputs      : Display a table with the head() and describe() of the data

# Creator      : Willy LAO
# Last update  : 17/05/2020
# Version      : Python 3

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

