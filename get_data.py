################################################################################################

#                 Structural Transformation and Network Effects
#                            R&R Economic Modelling

################################################################################################
# Marcos J Ribeiro
# Última atualização 20/03/2025
################################################################################################
# In this script I made some functions to get sea dataset


import os
import pandas as pd
import numpy as np
import plydata as pl
from pandas.api.types import CategoricalDtype


os.chdir(r"C:\Users\marco\Downloads\paper_EM")

#%%
################################################################################################
#                               GET WIOD
################################################################################################

def get_wiod(ano):

    ## get sectoral names
    set_n = (
             pd.read_excel('sea.xlsx', sheet_name = 'set_names') >>
                 pl.select('set_names','set_group') >>
                 pl.rename(description = 'set_names')
            )

    w14 = pd.read_pickle(f'wiot_{ano}.pkl')
    w14.set_index('Country', inplace = True)
    
    
    names = w14.index.drop_duplicates() # get country codes
    
    ###### Transform wiod in dictionary of dataframes by country     
    d = {}
    
    ii = 0
    var = 56
    
    for n in names:
        d[n] = w14.iloc[ (ii):(ii + var) , ii:(ii + var) ]
        ii += var
    
    # remove some countries
    del d['ROW']
    del d['TWN'] # nao tem a paridade do poder de compra para o twn por isso dropei
    del d['LUX']
    del d['MLT']
    del d['HRV']    
    
    ### get country code   
    ccode = list(d.keys())  
    ccode = np.array(ccode)
    
    # Groupy by sectors classification and collapsing by sum  
    for my_code in ccode:    
        
        d[my_code].columns = set_n.set_group
        d[my_code].index = set_n.set_group
        
        d[my_code] = d[my_code].groupby('set_group').sum().T
        #d[my_code] = d[my_code].T
        d[my_code] = d[my_code].groupby('set_group').sum().T
    
    return [d, ccode]


ccode = get_wiod(2000)[1]


#%%
##############################################################################################################################
#                              GET SOME DATASETS
##############################################################################################################################

## get Tornqvist index to VA_QI
index = pd.read_excel('index.xlsx')


## get sectoral names
set_n = (
         pd.read_excel('sea.xlsx', sheet_name = 'set_names') >>
             pl.select('set_names','set_group') >>
             pl.rename(description = 'set_names')
        )

### industries names
group_names = np.unique(np.array(set_n.set_group))


### get purchase power parity
ppp = pd.read_excel('ppp.xlsx', sheet_name= 'ppp')


### get SEA dataset
sea = (
        pd.read_excel('sea.xlsx', sheet_name = 'data1') >>
            pl.left_join(set_n, on = 'description') >>
            pl.group_by('set_group', 'country', 'variable') >>
            pl.summarise_if('is_numeric', np.sum) 
        )


## rowbind with tornqvist index
sea = pd.concat([sea, index])


###

# convert set_group column in order category
sea["set_group"] = sea["set_group"].astype(CategoricalDtype(categories=group_names, ordered=True))

# alphabetical order
sea = sea.sort_values(by=['country', 'variable', 'set_group'])


#%%
#######################################################
#                SEA dataset 
#######################################################

##### NON MONETARY VARIABLES
# slice SEA to get specified variable  and country
#sea_f('VA_index', code = 'USA', ano = 2011)

def sea_f(var, code, ano):
        
    data = sea.query('country == @code & variable == @var')[[f'year_{ano}']] 
    np.array(data.reset_index(drop = True, inplace = True) )
    
    return data

#%%

#######################################################
#          SEA dataset converted by PPP
#######################################################
# Na definição da OCDE, a taxa de PPP está expressa como 
# "moeda nacional por dólar americano" (ou seja, quantos 
# reais são necessários para comprar o equivalente a 1 dólar
# em poder de compra). Isso implica que a conversão correta do
# Gross Output para dólares deve ser feita dividindo o valor
# em reais pela taxa de PPP.


##### MONETARY VARIABLES

# this function get sea variable and country and divide by exchange rate
# slice SEA to get specified variable  and country
def sea_conv(var, code, ano):
    
    data = sea.query('country == @code & variable == @var')[[f'year_{ano}']] 
    data.reset_index(drop = True, inplace = True)
    
    data_conv = np.array(data/np.array(ppp.query('code == @code')[[f'year_{ano}']]))
        
    return data_conv

#sea_f(var = 'VA_QI', code = 'USA', ano = 2010)


#%%
#######################################################
#              COMPUTE SIGMA
#######################################################

def sigma_f(code, ano):
    lab = sea_f('LAB', code, ano)
    go = sea_f('GO', code, ano)  
    sigma = lab/go     ## go is in millions and lab is in millions
    sigma = np.array(sigma).reshape(i, 1)
        
    return sigma


#%%
#######################################################
#            COMPUTE TFP
#######################################################

def A_f(code, ano):
    va_torn = sea_f('VA_index', code, ano)
    empe = sea_f('EMPE', code, ano)
    A = np.array(va_torn/empe)
    
    return A


#%%
#######################################################
#              COMPUTE BETA
#######################################################

def beta_f(code, ano):

    data = get_wiod(ano)[0]
    data = data[code]
    data.reset_index(drop = True, inplace = True)
    data[data == 0] = 1e-4
    data = np.matrix(data)
    
    beta = np.divide(data, data.sum(1) )  ####
    #beta = beta.T
    
    return beta

#beta = beta_f(code = 'BRA', ano = 2013)

#%%
#######################################################
#       COMPUTE LABOR AND CONSUMPTION SHARES
#######################################################

def compute_dataset(code, ano):
        
    lab_sec_data = np.array(sea_f('EMP', code, ano)).reshape(i, 1)     
    sharelab_sec_data = lab_sec_data/np.sum(lab_sec_data) 
    
    GDP_sec_data =  np.array( sea_conv('VA', code, ano) ).reshape(i, 1)    
    share_cons_data = GDP_sec_data/np.sum(GDP_sec_data)
    
    go_sec = np.array( sea_conv('GO', code, ano) ).reshape(i, 1)
    share_go_data = go_sec/np.sum(go_sec)
    
    GDP_tot = np.repeat( np.sum(GDP_sec_data)/np.sum(lab_sec_data), i).reshape(i, 1)
    
    go_sec_usa = np.array( sea_conv('GO', 'USA', ano) ).reshape(i, 1)
      
    wgt_go = np.divide(go_sec, np.mean(go_sec_usa) ).reshape(i, 1)
    
    ii_sec = np.array( sea_conv('II', code, ano) ).reshape(i, 1)
    share_ii_data = np.divide(ii_sec, np.sum(ii_sec) )
    
    return [sharelab_sec_data, share_cons_data, share_go_data, GDP_tot, wgt_go, share_ii_data]




#%%

#######################################################
#              COMPUTE TAUS
#######################################################

# Tau_W
def tau_w_f(code, ano):
    sigma_usa = sigma_f('USA', ano)
    inv_sigma =  1/sigma_f(code, ano)
    tau_w = np.multiply(sigma_usa, inv_sigma) - 1
    
    return tau_w


# Tau_j
def tau_j_f(code, ano):
    beta_usa = beta_f('USA', ano)
    inv_beta = np.divide(1, beta_f(code, ano) )
    tau_j = np.multiply(beta_usa, inv_beta) - 1
    
    return tau_j
    
    
    
###########################################################################################################################
#                                 GET  FINAL USES 
###########################################################################################################################
# Here I get final uses dataset
# I use the same strategy that I used to get intermediate use dataset

# Final consumption expenditure by households
# Final consumption expenditure by non-profit organisations serving households (NPISH)
# Final consumption expenditure by government
# Gross fixed capital formation	Changes in inventories and valuables

def get_cons_f(year):
    
    #### GET DATA ####    
    uses = pd.read_pickle(f'uses_{year}.pkl')
    names = uses.index.drop_duplicates()

    #### Create a dictionary with final uses ####

    u = {}
 
    ii = 0
    var = 56
    var2 = 0
    
    for n in names:
        u[n] = uses.iloc[ (ii):(ii + var) , var2:(var2 + 5) ]
        ii += var
        var2 += 5
    
        
    del u['ROW']
    del u['TWN'] # nao tem a paridade do poder de compra para o twn por isso dropei
    del u['LUX']
    del u['MLT']
    del u['HRV']
    
    #### Variables Names ####
    u_names = np.array(['CONS_h', 'CONS_np', 'CONS_g', 'GFCF', 'INVEN'])
    
    for my_code in ccode:    
        
        u[my_code].columns = u_names
        u[my_code].index = set_n.set_group
        
        u[my_code] = u[my_code].groupby('set_group').sum()
    
    return u


#%%
##############################################################################################################################
#                                    Function to get final uses
##############################################################################################################################

def fim_use_f(code, year):
    u = get_cons_f(year)
    data = u[code] 
    #data = data/np.array(ppp.loc[ppp['code'] == code, f'year_{year}']).item()
    return data

    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



























