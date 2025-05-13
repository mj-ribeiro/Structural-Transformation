# #################################################################################################################################
# 
#                                         Structural Transformation and Network Effects
#                                                    R&R Economic Modelling
# 
# #################################################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 21/04/2025
# #################################################################################################################################
# #### In this script I made the functions of model from USA
# I will try to calibrate c_bar  using labor share 

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
num = lambda x: np.array2string(np.squeeze(x), formatter={'float_kind': lambda x: f"{x:.4f}"}, separator=", ").strip("[]")

#%%
######### PARAMETERS AND DATASETS ########################

## Parameters
year = 2010
i = 4
dt.i = 4
w = np.ones(i).reshape(i, 1)


#### Define A guess
def tfp_f(i):    
    A = np.ones(i).reshape(i, 1)
    return A


## Guess alpha and c_bar
def x1_f(i):
    c_bar = np.array([0.023, 0, 0.01, 0.011]).reshape(i, 1)
    return np.concatenate([c_bar.flatten()])


## Alternative Guess alpha and c_bar
def x2_f(i):
    c_bar = np.random.rand(i).reshape(i, 1)/100
    return np.concatenate([c_bar.flatten()])

#### Dataset #####

## beta
beta = dt.beta_f(code = 'USA', ano = year)

## sigma
sigma = dt.sigma_f(code = 'USA', ano = year)

## Households Consumption share (alpha)
cons = np.array(dt.fim_use_f(code = 'USA', year = year)['CONS_h'])
share_cons_d = (cons/np.sum(cons) ).reshape(i, 1)


#  labor share for a specific year
share_lab_d =  np.array([0.091, 0.332, 0.146, 0.431]).reshape(i, 1)


## guess alpha and tfp
x1 = x1_f(i)

L = 1 #np.sum(np.array(dt.sea_f(var = 'EMP', code = 'USA', ano = year) ) ) / 100 # change scale



#%%

############# Compute Equilibrium 
def equilibrium(x1, beta, sigma, alpha, L, i = i):
    
    # tfp
    A = np.ones(i).reshape(i, 1)
    
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
    
    ### Define c_bar
    c_bar = x1[:i].reshape(i, 1)
    #c_bar[2] = c_bar[3]
    
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

def out(x1):
    
    [p,
     C,
     B, 
     G,
     GDP,
     GDP_sec,
     share_cons_m,
     Li,
     share_lab_m] = equilibrium(x1 = x1, 
                                beta = beta, 
                                sigma = sigma, 
                                alpha = share_cons_d,
                                L = L)
    ii = np.multiply(p, np.sum(np.multiply(B, Li), axis = 0).reshape(i, 1) )
    go = np.multiply(p, np.multiply(G, Li) )
    #ii_sh = ii/np.sum(ii)
    
    print(f'\033[1;033mSectors: {dt.group_names}')
    print(f'c_bar: {num(x1[:4])}')
    print(f'Sectorial Labor: {num(Li)}')
    print(f'Total Labor: {num(np.sum(Li)) }')
    print(f'Labor share: {num(share_lab_m)}')
    print(f'Gross output: {num(go)}')
    print(f'Intermediate inputs: {num(ii)}')
    print(f'GDP sec: {num(GDP_sec)}')
    print(f'GDP: {num(GDP)}')
    print(f'Prices: {num(p)}')
    print(f'Share of consumption: {num(share_cons_m)}') 
    print(f'alpha: {num(share_cons_d)}')
    # restrição tem que dar zero
    print(f'Constraint: {num(np.multiply(G, Li) - np.sum(np.multiply(B, Li), axis = 0).reshape(i, 1) - C)}')


out(x1)

#%%

## callback to show the output of obj function
cc = 0
def callback(x1):    
    global cc
    cc += 1
    fobj = obj1(x1, beta, sigma, L, share_cons_d, share_lab_d)
    print(f'\033[1;033mObjetivo: {np.around(fobj, 8)}, iter: {cc}') 

 
#%%

### Objetive function
def obj1(x1, beta, sigma, L, share_cons_d, share_lab_d):    
    
    ## cbar
    x1[2] = 0 
    #x1[0] = np.min(x1)
     
    ## Run Model
    # [p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]
    result = equilibrium(x1 = x1,
                         beta = beta,
                         sigma = sigma, 
                         alpha = share_cons_d,
                         L = L)
    
    ## Labor Share model
    share_lab_m = np.round(result[8][:3], 3)
    #share_lab_m  = np.concatenate([share_lab_m, np.array([[1 - share_lab_m.sum()]])], axis=0)
    #share_lab_m = np.round(share_lab_m, 3)
    
    
    ## labor share data
    share_lab_d = np.round(share_lab_d[:3], 3)
    # share_lab_d  = np.concatenate([share_lab_d, np.array([[1 - share_lab_d.sum()]])], axis=0)
    # share_lab_d = np.round(share_lab_d, 3)

    ## Objective Function ####        
    obj = np.sum(np.power(np.divide(share_lab_m - share_lab_d, share_lab_d), 2)) 
      
    ## Constraints
    if np.any(result[5] < 0) or np.any(result[8] < 0):
        penalty = 1e5
        obj += penalty

    return obj



#%%
# guess
x1 = x2_f(i)

# Run function
res = obj1(x1 = x1,
        beta = beta,
        sigma = sigma, 
        L = L,
        share_cons_d = share_cons_d,
        share_lab_d = share_lab_d)

print(f"\033[1;033mResultado da função objetivo: {res:.4f}")


#%%

Bd = ( (0.0, 0.15), )*i  


cc = 0

#%%

my_iter = 500

sol = differential_evolution(
        obj1,
        bounds=Bd,
        args=(beta, sigma, L, share_cons_d, share_lab_d),
        strategy='randtobest1bin',      # Estratégia padrão, mas pode testar outras (ex: 'randtobest1bin')
        tol=1e-10,                 # Mais rigoroso
        mutation=(0.5, 1),         # Range de mutação, pode ajudar a escapar de ótimos locais
        recombination = 0.9,       # Alta recombinação favorece exploração
        maxiter=my_iter,           # Mais iterações para buscar melhor solução
        popsize=25,                # População maior para explorar melhor o espaço
        polish=True,               # Refinamento final c om método local
        disp=True,                 # Mostrar progresso
        updating='deferred',       # Pode ajudar com paralelismo
    )

## uses the differential evolution results on Nelder-Mead
sol = minimize(obj1,
               sol.x,
               method='Nelder-Mead', 
               bounds = Bd, 
               args=(beta, sigma, L, share_cons_d, share_lab_d),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter})


out(sol.x)



#%%

df = gen_res(sol.x)


def gen_res(x1):
    global share_lab_d, share_go_d
    
    res = []
    
    ## Run Model
    # [p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]
    result = equilibrium(x1 = x1,
                         beta = beta,
                         sigma = sigma, 
                         alpha = share_cons_d,
                         L = L)
    ## GO model share
    share_go_m = np.multiply(np.multiply(result[3], 1), result[0] ) #result[7]
    share_go_m = (share_go_m/np.sum(share_go_m))[:3]
    share_go_m  = np.concatenate([share_go_m, np.array([[1 - share_go_m.sum()]])], axis=0)
    share_go_m = np.round(share_go_m, 3)
    
    # GO data
    share_go_d = share_go_d[:3]
    share_go_d  = np.concatenate([share_go_d, np.array([[1 - share_go_d.sum()]])], axis=0)
    share_go_d = np.round(share_go_d, 3)
    
    
    ## Labor Share model
    share_lab_m = result[8][:3]
    share_lab_m  = np.concatenate([share_lab_m, np.array([[1 - share_lab_m.sum()]])], axis=0)
    share_lab_m = np.round(share_lab_m, 3)
    
    
    ## labor share data
    share_lab_d = share_lab_d[:3]
    share_lab_d  = np.concatenate([share_lab_d, np.array([[1 - share_lab_d.sum()]])], axis=0)
    share_lab_d = np.round(share_lab_d, 3)
    
    
    # save the results on list
    for ii, my_groups in enumerate(dt.group_names):
        res.append(
            {
            'code': 'USA',
            'year': year,
            'sectors': my_groups,
            'share_lab_m': share_lab_m[ii].item(),
            'share_lab_d': share_lab_d[ii].item(),
            'share_go_d': share_go_d[ii].item(),
            'share_go_m': share_go_m[ii].item(),
            }
        )
    
    df = pd.DataFrame(res)
    
    return df






#%%


def my_plot():
        
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (10, 7))
    year1 = [1950, 2010]
    
    ct = 0
    for ii in [0, 1]:
        for jj in [0, 1]:    
            code = dt.group_names[ct]
            
            x = np.array(df.query("sectors == @code ")['share_lab_m']).flatten()
            y = np.array(df.query("sectors == @code ")['share_lab_d'] ).flatten()
            
            ax[ii, jj].plot(year1, y, marker = 'o', label = 'data')
            ax[ii, jj].plot(year1, x, marker = 'v', label = 'model')
            ax[ii, jj].legend(loc="best")
            ax[ii, jj].set_title(code)
            ax[ii, jj].set_xticks(np.arange(1950, 2020, 20))
            
            ct = ct + 1
            
    fig.tight_layout()
    

my_plot()





#%%
#plt.clf()  # Limpa a figura atual






#%%


# #### Efficient objective function
# def obj1(x1, beta, sigma, L, share_lab_d_dict, A_dict):
    
#     x1[:i] = x1[:i] / np.sum(x1[:i]) # normalize o alpha
    
#     x1[5:7] = 0 # industry c_bar = 0
    
#     # Year List
#     years = [1950, 2010]
#     total_obj = 0
    
#     # Loop over years
#     for year in years:

#         # Get TFP
#         A = A_dict[year].reshape(i, 1) 
        
#         # Compute equilibrium 
#         result = equilibrium(x1 = x1, beta = beta, sigma = sigma, L = L, A = A)
        
#         # Get labor share from model
#         share_lab_m = result[8][:3]   # share_lab_m (vetor 4x1)

#         # Get labor share from data
#         share_lab_d = share_lab_d_dict[year].reshape(i, 1)[:3]

#         # Calculate objective
#         obj = np.sum(np.power(np.divide(share_lab_m - share_lab_d, share_lab_d), 2))
            
#         if np.any(np.min(result[5]) < 0) or np.any(np.min(result[8]) < 0): # if GDP_sec < 0 objective is +infinite 
#             obj = +np.inf

#         #print(f'obj: {obj}')

#         # Acumulate objetive function across years
#         total_obj += obj
    
#     return total_obj
























