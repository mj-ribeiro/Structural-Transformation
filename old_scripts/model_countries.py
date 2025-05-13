# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 12:47:14 2025

@author: marcos
"""

# #################################################################################################################################
# 
#                                         Structural Transformation and Network Effects
#                                                    R&R Economic Modelling
# 
# #################################################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 10/04/2025
# #################################################################################################################################
# #### In this script I made the functions of model WITH DISTORTIONS

#%%


import os
os.getcwd()
os.chdir(r"C:\Users\marco\Downloads\paper_EM")

import numpy as np
import pandas as pd
import get_data as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor

from scipy.optimize import minimize
from scipy.optimize import differential_evolution

np.set_printoptions(suppress=True)
np.set_printoptions(precision = 4)

# function to format output print of a vectors
num = lambda x: np.array2string(np.squeeze(x), formatter={'float_kind': lambda x: f"{x:.3f}"}, separator=", ").strip("[]")

#%%
######### PARAMETERS AND DATASETS ########################


#### Define A guess
def tfp_f(i):    
    A = np.random.rand(i).reshape(i, 1)+1
    A[3] = 1
    return A


## Parameters
year = 2014
i = 4
### Number of sectors
dt.i = 4
A = tfp_f(i)
country = 'BRA'

## USA c_bar
c_bar = np.array([0.0023, 0.0028, 0.    , 0.0061]).reshape(i, 1)

## USA alpha
alpha = np.array([0.0374, 0.0099, 0.2625, 0.6902]).reshape(i, 1)


# beta USA
beta = dt.beta_f(code = 'USA', ano = year) # use sigma and beta from USA

#sigma USA
sigma = dt.sigma_f(code = 'USA', ano = year)


# taus
tau_w = dt.tau_w_f(code = country, ano = year)
tau_j = dt.tau_j_f(code = country, ano = year)

# W = 1
w = np.ones(i).reshape(i, 1)


### Dataset #####

#  labor share for a specific year
share_lab_d = dt.compute_dataset(code= country, ano = year)[0]

# number of workers
L = 1 #np.sum(np.array(dt.sea_f(var = 'EMP', code = country, ano = year) ) ) / 100 # change scale




#Save some inputs of models in Excel file
# with pd.ExcelWriter('BRA_inputs.xlsx', engine='xlsxwriter') as writer:
#     pd.DataFrame(tau_w).to_excel(writer, sheet_name='tau_w')
#     pd.DataFrame(tau_j).to_excel(writer, sheet_name='tau_j')
#     pd.DataFrame(share_lab_d).to_excel(writer, sheet_name='labour_share')






#%%

############# Compute Equilibrium 
def equilibrium_code(A, c_bar, alpha, beta, sigma, L, tau_j, tau_w, i = i):
    
    A = A.reshape(i, 1)
    A[3] = 1
    
    ###### PRICES ########     
    sig1 = 1 - sigma 
    
    ## Matrix Phi
    P1 = - np.log(A) - np.multiply(sig1, np.log(sig1)) - np.multiply(sigma, np.log(sigma)) + \
                                    np.multiply(sigma, np.log(w) ) 
                            
    P2 = - np.multiply(sig1, np.sum( np.multiply(beta, np.log(beta) ), axis = 1).reshape(i, 1) )    
    
    P3 = np.multiply(sigma, np.log(1 + tau_w) ) + np.multiply(sig1, \
                                    np.sum( np.multiply(beta, np.log(1 + tau_j) ), axis = 1).reshape(i, 1))
    
    PHI = P1 + P2 + P3
    
    ### Matrix D multiplied by matrix sigma
    D = np.multiply(np.eye(i), sig1)
    
    ## Identity matrix
    I = np.eye(i)
    
    # Prices   
    log_p = np.dot(np.linalg.inv(I - np.dot(D, beta)), PHI)
    p = np.exp(log_p)

    #### Objects B and G ####        
    # Object B  (X/L)
    B = np.multiply(np.divide( (1+ tau_w), (1 + tau_j) ), \
                    np.multiply(np.multiply(np.divide(sig1, sigma), np.divide(beta, p.T) ), w) )
    
    # Object G  (Q/L)
    g1 = np.multiply(np.power(np.divide(sig1, sigma), sig1), np.power(np.multiply(w, (1 + tau_w) ) , sig1) )   
    g2 =  np.prod( np.power(np.divide(beta,  np.multiply((1 + tau_j), p.T)  ), \
                            np.multiply(beta, sig1) ), axis = 1).reshape(i, 1) 
    G = np.multiply(A, np.multiply(g1, g2))     
    
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

#[p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]

def out(my_A,  share_lab_d):
    
    [p,
    C,
    B, 
    G,
    GDP,
    GDP_sec,
    share_cons_m,
    Li,
    share_lab_m] = equilibrium_code(
                      A = my_A, 
                      c_bar = c_bar, 
                      alpha = alpha,
                      beta = beta,
                      sigma = sigma, 
                      L = L,
                      tau_j = tau_j,
                      tau_w = tau_w
                      )
                                     
    share_lab_m1 = share_lab_m[:3]
    share_lab_d1 = share_lab_d[:3]
     
    obj = np.sum(np.power(np.divide(share_lab_m1 - share_lab_d1, share_lab_d1), 2))
    
    if np.any(C < 0) or np.any(Li < 0):
        obj = 1e6
   
    print(f'\033[1;33mObjective Function: {num(obj)}')
    print(f'Sectors: {dt.group_names}')
    print(f'TFP: {num(my_A)}')
    # print(f'c_bar: {num(c_bar)}')
    print(f'Sectorial Labor: {num(Li)}')
    print(f'Total Labor: {num(np.sum(Li)) }')
    print(f'Labor share MODEL: {num(share_lab_m)}')
    print(f'Labor share DATA: {num(share_lab_d)}')
    print(f'GDP sec: {num(GDP_sec)}')
    print(f'GDP: {num(GDP)}')
    print(f'Prices: {num(p)}')
    print(f'Share of consumption: {num(share_cons_m)}')    
    # restrição tem que dar zero
    print(f'Constraint: {num(np.multiply(G, Li) - np.sum(np.multiply(B, Li), axis = 0).reshape(i, 1) - C)}')


A = np.array([1, 1, 1, 1]).reshape(i, 1)

out(A, share_lab_d)


#%%

#[p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]

### objetive function
def obj1(A, c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d):
    
    A[3] = 1 # normalize productivity
    
    ## run equilibrium
    result = equilibrium_code(A = A, 
                      c_bar = c_bar, 
                      alpha = alpha,
                      beta = beta,
                      sigma = sigma, 
                      L = L,
                      tau_j = tau_j,
                      tau_w = tau_w)
    
    ## labor share of model        
    share_lab_m = result[8][:3]
    share_lab_d = share_lab_d[:3]
    
    obj = np.sum(np.power(np.divide(share_lab_m - share_lab_d, share_lab_d), 2))

    if np.any(result[5] < 0) or np.any(result[8] < 0):
        obj = 1e6

    return obj


#%%

## TESTS

A = tfp_f(i)

out(A, share_lab_d)



#%%


## callback to show the output of obj function
cc = 0
def callback(A):    
    global cc
    cc += 1
    fobj = obj1(A,
                 c_bar,
                 alpha,
                 beta,
                 sigma,
                 L,
                 tau_j,
                 tau_w,
                 share_lab_d)
    print(f'\033[1;033mObjetivo: {np.around(fobj, 8)}, iter: {cc}') 


#%%


Bd = ( (0.1, 500), )*i  
my_iter = 5000

cc=0
##  Nelder-Mead
sol = minimize(obj1,
               A.flatten(),
               method='Nelder-Mead', 
               bounds = Bd, 
               args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter})



#%%

Bd = ( (0.001, 100), )*i  


sol = differential_evolution(
    obj1,
    bounds=Bd,
    args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d),
    strategy='randtobest1bin',      # Estratégia padrão, mas pode testar outras (ex: 'randtobest1bin')
    tol=1e-10,                 # Mais rigoroso
    mutation=(0.5, 1),         # Range de mutação, pode ajudar a escapar de ótimos locais
    recombination = 0.9,         # Alta recombinação favorece exploração
    maxiter=500,              # Mais iterações para buscar melhor solução
    popsize=25,                # População maior para explorar melhor o espaço
    polish=True,               # Refinamento final com método local
    disp=True,                 # Mostrar progresso
    updating='deferred',       # Pode ajudar com paralelismo
)


##  Nelder-Mead
sol = minimize(obj1,
               sol.x,
               method='Nelder-Mead', 
               bounds = Bd, 
               args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter})


out(sol.x, share_lab_d)


#%%

#### RUN FOR ALL COUNTRIES ####


Bd = ( (0.001, 500), )*i  
res = []

for country in dt.ccode:
    
    print(f'\033[1;033m---'*11)       
    print(f'Country: {country}')
    
    # taus
    tau_w = dt.tau_w_f(code = country, ano = year)
    tau_j = dt.tau_j_f(code = country, ano = year)
      
    #  labor share for a specific year
    share_lab_d = dt.compute_dataset(code= country, ano = year)[0]
    
    # total labor
    L = 1#np.sum(np.array(dt.sea_f(var = 'EMP', code = country, ano = year) ) ) / 1e5 # change scale

    
    print(f'\033[1;033mRun Differential Evolution!')
    ## Run solver
    sol = differential_evolution(
            obj1,
            bounds=Bd,
            args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d),
            strategy='randtobest1bin',      # Estratégia padrão, mas pode testar outras (ex: 'randtobest1bin')
            tol=1e-10,                 # Mais rigoroso
            mutation=(0.5, 1),         # Range de mutação, pode ajudar a escapar de ótimos locais
            recombination = 0.9,         # Alta recombinação favorece exploração
            maxiter=500,              # Mais iterações para buscar melhor solução
            popsize=25,                # População maior para explorar melhor o espaço
            polish=True,               # Refinamento final com método local
            #disp=True,                 # Mostrar progresso
            updating='deferred'       # Pode ajudar com paralelismo
        )
      
    print(f'\033[1;033mPolishing solution.')
    
    ## force traditional services equal one
    sol.x[3] = 1
    
    ##  Nelder-Mead
    sol = minimize(
            obj1,
            sol.x,
            method='Nelder-Mead', 
            bounds = Bd, 
            args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d),
            #callback = callback, 
            tol = 1e-8,
            options={'maxiter': my_iter})
    

    #[p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]
    print(f'\033[1;033mObjective: {num(sol.fun)}')
    
    print(f'\033[1;033mRun Model!')    
    ## run equilibrium
    model = equilibrium_code(A = sol.x, 
                      c_bar = c_bar, 
                      alpha = alpha,
                      beta = beta,
                      sigma = sigma, 
                      L = L,
                      tau_j = tau_j,
                      tau_w = tau_w)
    
    print(f'TFP: {num(sol.x)}')
    
    print(f'\033[1;033mSaving results!')
    # save the results on list
    for ii, my_groups in enumerate(dt.group_names):
        res.append(
            {
            'code': country,
            'year': year,
            'sectors': my_groups,
            'share_lab_m': model[8][ii].item(),
            'share_lab_d': share_lab_d[ii].item(),
            'tfp': sol.x[ii].item(),       
            'obj': sol.fun,
            'success': sol.success
            }
        )


df = pd.DataFrame(res)

df.to_excel('results.xlsx')


















