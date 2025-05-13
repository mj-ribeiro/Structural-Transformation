# 
# #################################################################################################################################
# 
#                                         Structural Transformation and Network Effects
#                                                    R&R Economic Modelling
# 
# #################################################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 02/04/2025
# #################################################################################################################################
# #### In this script I made the functions of model 

#%%

import os
os.getcwd()
os.chdir(r"C:\Users\marco\Downloads\paper_EM")

import numpy as np
import pandas as pd
import get_data as dt
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import differential_evolution

np.set_printoptions(suppress=True)
np.set_printoptions(precision = 4)

# function to format output print of a vectors
num = lambda x: np.array2string(np.squeeze(x), formatter={'float_kind': lambda x: f"{x:.3f}"}, separator=", ").strip("[]")

#%%
######### PARAMETERS AND DATASETS ########################

## GET DATA SET (TFP and Labor Share)
tfp = pd.read_excel('data_usa_tfp.xlsx')
labsh = pd.read_excel('data_usa_labsh.xlsx')


## Transform the data in dictionary
A_dict = {year: tfp[year].values for year in tfp.columns}
share_lab_d_dict = {year: labsh[year].values for year in labsh.columns}

## Convert keys of dictionaries in numeric
A_dict = {int(k): v for k, v in A_dict.items()}
share_lab_d_dict = {int(k): v for k, v in share_lab_d_dict.items()}


#### Define A guess
def tfp(i):    
    A = np.ones(i).reshape(i, 1)
    return A


## Guess alpha and c_bar
def x1_f(i):
    alpha = np.array([0.2, 0.4, 0.2, 0.2]).reshape(i, 1)
    c_bar = np.array([0.023, 0.031, 0.01, 0.011]).reshape(i, 1)
    return np.concatenate([alpha.flatten(), c_bar.flatten()])


## Alternative Guess alpha and c_bar
def x2_f(i):
    alpha = np.array([0.1, 0.2, 0.4, 0.3])#np.random.rand(i).reshape(i, 1)
    alpha = alpha/np.sum(alpha)

    c_bar = np.random.rand(i).reshape(i, 1)/100
    return np.concatenate([alpha.flatten(), c_bar.flatten()])


## Parameters
year = 2010
i = 4
A = tfp(i)
beta = dt.beta_f(code = 'USA', ano = year)
sigma = dt.sigma_f(code = 'USA', ano = year)
w = np.ones(i).reshape(i, 1)

### sector number
dt.i = 4

### Dataset #####

#  labor share for a specific year
share_lab_d = dt.compute_dataset(code= 'USA', ano = year)[0]

# share of consumption for a specific year
share_cons_d = dt.compute_dataset(code = 'USA', ano = year)[1]

## guess alpha and c_bar
x1 = x1_f(i)

L = 1 #np.sum(np.array(dt.sea_f(var = 'EMP', code = 'USA', ano = year) ) ) / 100 # change scale


#%%

############# Compute Equilibrium 
def equilibrium(x1, beta, sigma, L, A = A, i = i):
    
    ###### PRICES ########     
    sig1 = 1 - sigma 
    
    ## Matrix Phi
    P1 = - np.log(A) - np.multiply(sig1, np.log(sig1)) - np.multiply(sigma, np.log(sigma)) + \
                            np.multiply(sigma, np.log(w) ) 
    P2 = - np.multiply(sig1, np.sum( np.multiply(beta, np.log(beta) ), axis = 1).reshape(i, 1) )    
    
    PHI = P1 + P2 
    
    ### Matrix D multiplied by matrix sigma
    D = np.multiply(np.eye(i), sig1)
    
    ## Identity matrix
    I = np.eye(i)
    
    # Prices   
    log_p = np.dot(np.linalg.inv(I - np.dot(D, beta)), PHI)
    p = np.exp(log_p)

    #### Objects B and G ####        
    # Object B  (X/L)
    B =  np.multiply(np.multiply(np.divide(sig1, sigma), np.divide(beta, p.T) ), w)   

    # Object G  (Q/L)
    g1 = np.multiply(np.power(np.divide(sig1, sigma), sig1), np.power(w, sig1) )   
    g2 =  np.prod( np.power(np.divide(beta, p.T), np.multiply(beta, sig1) ), axis = 1).reshape(i, 1) 
    G = np.multiply(A, np.multiply(g1, g2))     
    
    ### Define alpha and c_bar
    c_bar = x1[i:].reshape(i, 1); alpha = x1[:i].reshape(i, 1)

    alpha = alpha/np.sum(alpha) 
    
    

    ##### CONSUMPTION ######     
    C = c_bar +  np.multiply(np.divide(alpha, p), (np.multiply(w, L) - p.T@c_bar) )
    
    ####### LABOUR  #######
    
    #### Auxiliar Matrices ##### 
    # B divide by G
    B_til = np.divide(B, G.T)
    C_til = np.divide(C, G)
    
    ## LHS
    M = I - B_til.T
    
    #### Labor
    ## Calcular L_i 
    Li = np.dot(np.linalg.inv(M), C_til)
    
    ## labor share
    share_lab = Li/np.sum(Li)       
    
    #### GDP ####   
    GDP = np.sum(np.multiply(p, C)) 
                                                        
    #### Sectoral GDP ####
    GDP_sec = np.multiply(p, C)
    
    #### share of consumption on GDP
    share_cons = GDP_sec/GDP
        
    return [p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]



#%%

def out(x1, my_A):
    
    [p,
     C,
     B, 
     G,
     GDP,
     GDP_sec,
     share_cons_m,
     Li,
     share_lab_m] = equilibrium(x1 = x1, beta = beta, sigma = sigma, L = L, A = my_A)
    
    obj = np.sum( np.power(np.divide((share_cons_m - share_cons_d), share_cons_m), 2) ) +   \
    np.sum( np.power(np.divide(( share_lab_m -  share_lab_d ),  share_lab_m ), 2) )
    print(f'\033[1;33m Objective Function: {num(obj)}')
    print(f'Sectors: {dt.group_names}')
    print(f'Sectorial Labor: {num(Li)}')
    print(f'Total Labor: {num(np.sum(Li)) }')
    print(f'Labor share: {num(share_lab_m)}')
    print(f'GDP sec: {num(GDP_sec)}')
    print(f'GDP: {num(GDP)}')
    print(f'Prices: {num(p)}')
    print(f'Share of consumption: {num(share_cons_m)}')    
    # restrição tem que dar zero
    print(f'Constraint: {num(np.multiply(G, Li) - np.sum(np.multiply(B, Li), axis = 0).reshape(i, 1) - C)}')


#%%
my_A = np.array([1, 1, 1, 1]).reshape(i, 1)

out(x2_f(i), my_A)


#%%

## callback to show the output of obj function
cc = 0
def callback(x1):    
    global cc
    cc += 1
    fobj = obj1(x1, beta, sigma, L, share_lab_d_dict, A_dict)
    print(f'\033[1;033mObjetivo: {np.around(fobj, 8)}, iter: {cc}') 


#%%

#### Efficient objective function

def obj1(x1, beta, sigma, L, share_lab_d_dict, A_dict):
    
    x1[:i] = x1[:i] / np.sum(x1[:i]) # normalizar o alpha
    
    # Year List
    years = range(1950, 2020, 10)
    total_obj = 0
    
    # Loop over years
    for year in years:

        # Get TFP
        A = A_dict[year].reshape(i, 1) # A0 * e^{r (t-1960)} 

        #A = A*A0 # calibrar o A0
        
        # Compute equilibrium 
        result = equilibrium(x1=x1, beta=beta, sigma=sigma, L=L, A=A)
        
        # Get labor share from model
        share_lab_m = result[8][:3]   # share_lab_m (vetor 4x1)

        # Get labor share from data
        share_lab_d = share_lab_d_dict[year].reshape(i, 1)[:3]

        # Calculate objective
        obj = np.sum(np.power(np.divide(share_lab_m - share_lab_d, share_lab_d), 2))


        # Penalties
        if np.sum(x1[:i]) != 1:
            obj = +np.inf
        
        if np.min(result[5]) < 0: # if GDP_sec < 0 objective is +infinite 
            obj = +np.inf

        #print(f'obj: {obj}')

        # Acumulate objetive function across years
        total_obj += obj
    
    return total_obj



#%%

# guess
x1 = x2_f(i)

# Run function
res = obj1(x1 = x1,
    beta = beta,
    sigma = sigma, 
    L = L,
    share_lab_d_dict = share_lab_d_dict,
    A_dict = A_dict)

print(f"\033[1;033mResultado da função objetivo: {res:.4f}")


#%%
# chute: array([0.1   , 0.2   , 0.4   , 0.3   , 0.0011, 0.006 , 0.0025, 0.0087])
#array([0.0533, 0.2141, 0.13  , 0.6026, 0.003 , 0.0038, 0.0005, 0.009 ])
# 1.17276231

# chute 2: array([0.0533, 0.2141, 0.13  , 0.6026, 0.003 , 0.0038, 0.0005, 0.009 ])
# array([0.0534, 0.1912, 0.1495, 0.6058, 0.0028, 0.0034, 0.0005, 0.0097])
# 1.14927829


## Guess alpha and c_bar
def x1_f(i):
    alpha = np.array([0.2, 0.41, 0.12, 0.2]).reshape(i, 1)
    c_bar = np.array([0.91, 0.4121, 0.51, 0.51]).reshape(i, 1)
    return np.concatenate([alpha.flatten(), c_bar.flatten()])


#x1 = x2_f(i)

# chute otimizado
x1 = np.array([0.0342, 0.1026, 0.0778, 0.4241, 0.0026, 0.0028, 0.0001, 0.0018])

Bd = ((1e-6, 0.6), )*i + ( (0.0001, 0.01), )*i  

my_iter = 500
cc=0

sol = minimize(obj1,
               x1,
               method='Nelder-Mead', 
               bounds = Bd, 
               args= (beta, sigma, L, share_lab_d_dict, A_dict),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter})


#%%
#sol = differential_evolution(obj1, bounds=Bd, tol=1e-10, disp=True, maxiter=1000, popsize=200)

Bd = ((1e-6, 0.6), )*i + ( (0.0001, 0.1), )*i  
my_iter = 500

sol = differential_evolution(
    obj1,
    bounds=Bd,
    args=(beta, sigma, L, share_lab_d_dict, A_dict),
    tol=1e-8,
    maxiter = my_iter,    
    popsize = 20, 
    disp = True)



## uses the differential evolution results on Nelder-Mead
sol = minimize(obj1,
               sol.x,
               method='Nelder-Mead', 
               bounds = Bd, 
               args= (beta, sigma, L, share_lab_d_dict, A_dict),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter})


# 1.10867
#result: array([0.0535, 0.1607, 0.1218, 0.664 , 0.0026, 0.0028, 0.0001, 0.0018])

#        array([0.0535, 0.1607, 0.1218, 0.664 , 0.0026, 0.0028, 0.0001, 0.0018])



#%%

x1 = sol.x
res = []
years = range(1950, 2020, 10)

# Loop over years
for year in years:
    print(f'\033[1;033mYear:{year}')

    # Get TFP
    A = A_dict[year].reshape(i, 1)

    # Compute equilibrium 
    result = equilibrium(x1=x1, beta=beta, sigma=sigma, L=L, A=A)

    # Get labor share from model
    share_lab_m = result[8]  # share_lab_m (vetor 4x1)

    # Get labor share from data
    share_lab_d = share_lab_d_dict[year].reshape(i, 1)

     # save the results on list
    for ii, my_groups in enumerate(dt.group_names):
        res.append(
            {
            'code': 'USA',
            'year': year,
            'sectors': my_groups,
            'share_lab_m': share_lab_m[ii].item(),
            'share_lab_d': share_lab_d[ii].item(),
            
            }
        )

df = pd.DataFrame(res)


#%%
# [p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]

my_A = np.array([1, 1, 1, 120]).reshape(i, 1)

# Compute equilibrium 
result = equilibrium(x1=x1, beta=beta, sigma=sigma, L=L, A=my_A.reshape(i, 1))

# Get labor share from model
share_lab_m = result[8]  # share_lab_m (vetor 4x1)

Li = result[8]  # 


print(Li)



#%%

def my_plot(code):
    
    x = np.array(df.query("sectors == @code ")['share_lab_m']).flatten()
    y = np.array(df.query("sectors == @code ")['share_lab_d'] ).flatten()
    year1 = np.array([jj for jj in range(1950, 2020, 10)])

    plt.plot(year1, y, label = 'data')
    plt.plot(year1, x, label = 'model')
    plt.legend(loc="best")



print('oi')


#%%
plt.clf()  # Limpa a figura atual
my_plot('Traditional Services')

#share_lab_d_dict


A_dict














#%%
# Número de tentativas
n_trials = 10
best_result = None
best_value = np.inf


for ii in range(n_trials):
    print(f'Iteration: {ii}')

    x0 = x2_f(i)
    result = minimize(obj1, x0, method='Nelder-Mead', bounds=Bd, 
                      args=(beta, sigma, L, share_lab_d_dict, A_dict),
                      tol=1e-20, options={'maxiter': int(2e3)})
    if result.fun < best_value:
        best_value = result.fun
        best_result = result

print("Melhor valor encontrado:", best_value)
print("Solução:", best_result.x)
