
# #################################################################################################################################
# 
#                                Structural Transformation and Network Effects
#                                          R&R Economic Modelling
# 
# #################################################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 14/05/2025
# #################################################################################################################################
# #### In this script I made the functions of model WITH DISTORTIONS
# I will calibrate TFP using gross output weighted by the average sectoral GO from USA




#%%

import os
os.getcwd()
os.chdir(r"C:\Users\marco\Downloads\paper_EM")

import numpy as np
import pandas as pd
import get_data as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    A[0] = 1
    return A


## Parameters
year = 2014
i = 4
### Number of sectors
dt.i = 4
A = tfp_f(i)
country = 'CYP'

## USA c_bar (I calibrated)
c_bar = np.array([0.0023, 0.0027, 0.0000, 0.0066]).reshape(i, 1)


## Households Consumption share (alpha)
cons = np.array(dt.fim_use_f(code = country, year = year)['CONS_h'])
alpha = (cons/np.sum(cons) ).reshape(i, 1)


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


# Weighted Gross output for a specific year
wgt_go_d = dt.compute_dataset(code = country, ano = year)[4]


#%%

############# Compute Equilibrium 
def equilibrium_code(x1, c_bar, alpha, beta, sigma, L, tau_j, tau_w, i = i):
    
    A = x1.reshape(i, 1)
    
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

    ## normalize prices
    #p = p/p[0]
    
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

def out(my_A,  share_lab_d, wgt_go_d):
    
    [p,
    C,
    B, 
    G,
    GDP,
    GDP_sec,
    share_cons_m,
    Li,
    share_lab_m] = equilibrium_code(
                      x1 = my_A, 
                      c_bar = c_bar, 
                      alpha = alpha,
                      beta = beta,
                      sigma = sigma, 
                      L = L,
                      tau_j = tau_j,
                      tau_w = tau_w

                         )
       
    go_usa = np.array([0.4979, 1.5412, 0.5382, 0.9384]).reshape(i, 1)
                               
    ## GO model share
    go_m = np.multiply(G, Li) #
    wgt_go_m = (go_m/ np.mean(go_usa))
    
    
    eq5 = np.divide(np.multiply(np.multiply(sigma, p), np.multiply(G, Li) ), (1+tau_w) )
    
    print(f'\033[1;033mSectors: {dt.group_names}')
    print(f'TFP: {num(my_A)}')
    print(f'Equation 5: {num(eq5)}')
    print(f'Sectorial Labor: {num(Li)}')
    print(f'Total Labor: {num(np.sum(Li)) }')
    print(f'Labor share MODEL: {num(share_lab_m )}')
    print(f'Labor share DATA: {num(share_lab_d)}')
    print(f'Wgt Gross output MODEL: {num(wgt_go_m)}')
    print(f'Wgt Gross output DATA: {num(wgt_go_d)}')
    print(f'GDP sec: {num(GDP_sec)}')
    print(f'GDP: {num(GDP)}')
    print(f'Prices: {num(p)}')
    #print(f'Share of consumption: {num(share_cons_m)}')    
    print(f'Amount Consumed: {num(C)}')
    print(f'Amount of Intermediate Inputs: {num(np.sum(np.multiply(B, Li), axis = 0))}')    
    print(f'Amount of Gross Output: {num(np.multiply(G, Li))}')    
    
    # restrição tem que dar zero
    print(f'Constraint: {num(np.multiply(G, Li) - np.sum(np.multiply(B, Li), axis = 0).reshape(i, 1) - C)}')


A = np.array([0.9775, 1.3271, 1.0949, 0.7789]).reshape(i, 1)

out(A, share_lab_d, wgt_go_d)


#%%

#[p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]

### objetive function
def obj1(x1, c_bar, alpha, beta, sigma, L, tau_j, tau_w, wgt_go_d):
        
    ## run equilibrium
    result = equilibrium_code(x1 = x1, 
                      c_bar = c_bar, 
                      alpha = alpha,
                      beta = beta,
                      sigma = sigma, 
                      L = L,
                      tau_j = tau_j,
                      tau_w = tau_w)
    
    ## GO model relative to USA
    
    # GO model
    go_m = np.multiply(result[3], result[7])
    
    # GO USA
    go_usa = np.array([0.4979, 1.5412, 0.5382, 0.9384])
                                    
    wgt_go_m = np.divide(go_m, np.sum(go_usa) ) # divide by total US
      
    ## Objective Function ####        
    obj = np.sum(np.power(np.divide(wgt_go_m - wgt_go_d, wgt_go_d ), 2)) 
          
    if np.any(result[5] < 0) or np.any(result[8] < 0):
        obj = 1e6

    return obj


#%%


## callback to show the output of obj function
cc = 0
def callback(x1):    
    global cc
    cc += 1
    fobj = obj1(x1,
                 c_bar,
                 alpha,
                 beta,
                 sigma,
                 L,
                 tau_j,
                 tau_w,
                 wgt_go_d)
    print(f'\033[1;033mObjetivo: {np.around(fobj, 8)}, iter: {cc}') 


#%%


Bd = ( (0.000095, 10), )*i  
my_iter = 5000

x1 = tfp_f(i)

cc=0
##  Nelder-Mead
sol = minimize(obj1,
               x1.flatten(),
               method='Nelder-Mead', 
               bounds = Bd, 
               args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, wgt_go_d),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter})

    
#%%

#  labor share for a specific year
#share_lab_d = (np.ones(i)/4).reshape(i, 1)

    
Bd = ( (0.001, 5), )*i  

sol = differential_evolution(
    obj1,
    bounds=Bd,
    args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, wgt_go_d),
    strategy='randtobest1bin',      # Estratégia padrão, mas pode testar outras (ex: 'randtobest1bin')
    tol=1e-10,                 # Mais rigoroso
    mutation=(0.5, 1),         # Range de mutação, pode ajudar a escapar de ótimos locais
    recombination = 0.9,         # Alta recombinação favorece exploração
    maxiter=600,              # Mais iterações para buscar melhor solução
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
               args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, wgt_go_d),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter*10})


out(sol.x, share_lab_d, wgt_go_d) 


#%%

#### RUN FOR ALL COUNTRIES ####

res = []
go_usa = np.array([0.4979, 1.5412, 0.5382, 0.9384]) # gross output USA
L = 1
Bd = ( (0.000095, 10), )*i  
my_iter = 5000

 
for country in dt.ccode:
    
    print('\033[1;033m---'*11)       
    print(f'Country: {country}')
    
    # taus
    tau_w = dt.tau_w_f(code = country, ano = year)
    tau_j = dt.tau_j_f(code = country, ano = year)
      
    #  labor share for a specific year
    share_lab_d = dt.compute_dataset(code= country, ano = year)[0]
        
    # Weighted Gross output for a specific year
    wgt_go_d = dt.compute_dataset(code = country, ano = year)[4]
    
    ## Households Consumption share (alpha)
    cons = np.array(dt.fim_use_f(code = country, year = year)['CONS_h'])
    alpha = (cons/np.sum(cons) ).reshape(i, 1)
    
    print('\033[1;033mRun Differential Evolution!')
    ## Run solver
    sol = differential_evolution(
            obj1,
            bounds=Bd,
            args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, wgt_go_d),
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
      
    print('\033[1;033mPolishing solution.')
    
    ##  Nelder-Mead
    sol = minimize(
            obj1,
            sol.x,
            method='Nelder-Mead', 
            bounds = Bd, 
            args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, wgt_go_d),
            #callback = callback, 
            tol = 1e-8,
            options={'maxiter': my_iter})
    
    #[p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]
    print(f'\033[1;033mObjective: {num(sol.fun)}')
    
    print('\033[1;033mRun Model!')    
    ## run equilibrium
    model = equilibrium_code(x1 = sol.x, 
                      c_bar = c_bar, 
                      alpha = alpha,
                      beta = beta,
                      sigma = sigma, 
                      L = L,
                      tau_j = tau_j,
                      tau_w = tau_w
                      )
    
    print(f'TFP: {num(sol.x)}')
    
    ## GO model share
    go_m = np.multiply(model[3], model[7]) #
                               
    wgt_go_m = (go_m/ np.sum(go_usa) )
    
    ## labor
    tot_lab = np.sum(model[7])
    
    print('\033[1;033mSaving results!')
    # save the results on list
    for ii, my_groups in enumerate(dt.group_names):
        res.append(
            {
            'code': country,
            'year': year,
            'sectors': my_groups,
            'wgt_go_m': wgt_go_m[ii].item(),
            'wgt_go_d': wgt_go_d[ii].item(),
            'share_lab_m': model[8][ii].item(),
            'share_lab_d': share_lab_d[ii].item(),
            'tfp': sol.x[ii].item(),       
            'GDP': tot_lab, 
            'obj': sol.fun,
            'success': sol.success
            }
        )


df = pd.DataFrame(res)

df.to_excel('results_go_wgt.xlsx', index = False)


















