
# #################################################################################################################################
# 
#                                Structural Transformation and Network Effects
#                                          R&R Economic Modelling
# 
# #################################################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 05/05/2025
# #################################################################################################################################
# #### In this script I made the functions of model WITH DISTORTIONS
# I will calibrate TFP using gross labor share and GDP per worker at US prices


#%%


import os
os.getcwd()
os.chdir(r"C:\Users\marco\Downloads\paper_EM")

import numpy as np
import pandas as pd
import get_data as dt
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

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
    return A


## Parameters
year = 2014
i = 4
### Number of sectors
dt.i = 4
A = tfp_f(i)
country = 'IND'

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


# share of gross output for a specific year
share_go_d = dt.compute_dataset(code = country, ano = year)[2]


## GDP per worker relative to US for a specific year ##

# GDP per worker - USA
gdp_tot_usa = np.sum(np.sum(dt.fim_use_f(code = 'USA', year = year), axis = 1) )
emp_tot_usa = np.sum(np.array(dt.sea_f('EMP', code = 'USA', ano = 2014)))

gdp_pw_usa = gdp_tot_usa/emp_tot_usa

# GDP per worker - Other countries
gdp_tot = np.sum(np.sum(dt.fim_use_f(code = country, year = year), axis = 1) )
emp_tot = np.sum(np.array(dt.sea_f('EMP', code = country, ano = 2014)))

gdp_pw = gdp_tot/emp_tot

## relative GDP per worker
rel_gdp_pw_d = gdp_pw/gdp_pw_usa





#%%

############# Compute Equilibrium 
def equilibrium_code(x1, c_bar, alpha, beta, sigma, L, tau_j, tau_w, i = i):
   
    ## define TFP
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

def out(my_A,  share_lab_d, share_go_d, rel_gdp_pw_d):
    
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
                                     
    ## GO model share
    share_go_m = np.multiply(G, Li) #
    share_go_m = (share_go_m/np.sum(share_go_m))
    
    ## usa prices
    p_usa = np.array([154.5245, 124.9816, 79.8704, 36.8712]).reshape(i, 1)
    
    ## GDP at USA prices
    rel_gdp_pw_m = np.sum( np.divide(np.multiply(C, p_usa), Li) ) 
   
    print(f'\033[1;033mSectors: {dt.group_names}')
    print(f'TFP: {num(my_A)}')
    print(f'c_bar: {num(c_bar)}')
    print(f'Sectorial Labor: {num(Li)}')
    print(f'Total Labor: {num(np.sum(Li)) }')
    print(f'Labor share MODEL: {num(share_lab_m)}')
    print(f'Labor share DATA: {num(share_lab_d)}')
    print(f'Gross output MODEL: {num(share_go_m)}')
    print(f'Gross output DATA: {num(share_go_d)}')
    print(f'GDP at US prices MODEL: {rel_gdp_pw_m}')
    print(f'GDP at US prices DATA: {rel_gdp_pw_d}')
    print(f'GDP sec: {num(GDP_sec)}')
    print(f'GDP: {num(GDP)}')
    print(f'Prices: {num(p)}')
    print(f'Share of consumption: {num(share_cons_m)}')    
    # restrição tem que dar zero
    print(f'Constraint: {num(np.multiply(G, Li) - np.sum(np.multiply(B, Li), axis = 0).reshape(i, 1) - C)}')


A = np.array([1, 1, 0.5, 1]).reshape(i, 1)

out(A, share_lab_d, share_go_d, rel_gdp_pw_d)


#%%

#[p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]

### objetive function
def obj1(x1, c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d, share_go_d, rel_gdp_pw_d):
    
    
    ## run equilibrium
    result = equilibrium_code(x1 = x1, 
                      c_bar = c_bar, 
                      alpha = alpha,
                      beta = beta,
                      sigma = sigma, 
                      L = L,
                      tau_j = tau_j,
                      tau_w = tau_w)
    ## usa prices
    p_usa = np.array([154.5245, 124.9816, 79.8704, 36.8712]).reshape(i, 1)
    
    ## GDP at USA prices
    rel_gdp_pw_m = np.sum( np.divide(np.multiply(result[1], p_usa), result[7]) ) 
    
    ## GO model share
    share_go_m = np.multiply(result[3], result[7]) #
    share_go_m = (share_go_m/np.sum(share_go_m))[:3]
    share_go_m  = np.concatenate([share_go_m, np.array([[1 - share_go_m.sum()]])], axis=0)
    share_go_m = np.round(share_go_m, 3)
    
    # GO data
    share_go_d = share_go_d[:3]
    share_go_d  = np.concatenate([share_go_d, np.array([[1 - share_go_d.sum()]])], axis=0)
    share_go_d = np.round(share_go_d, 3)
    
    ## labor share of model        
    share_lab_m = result[8][:3]
    share_lab_d = share_lab_d[:3]
    
    ## Objective Function ####        
    #obj = np.sum(np.power(np.divide(share_go_m - share_go_d, share_go_d), 2)) 
          
    obj = np.sum(np.power(np.divide(share_lab_m - share_lab_d, share_lab_d), 2)) + \
            np.power(np.divide( (rel_gdp_pw_m - rel_gdp_pw_d), rel_gdp_pw_d), 2)

    if np.any(result[5] < 0) or np.any(result[8] < 0):
        obj = 1e6

    return obj


#%%

## TESTS

x1 = tfp_f(i)

out(x1, share_lab_d, share_go_d, rel_gdp_pw_d)



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
                 share_lab_d, 
                 share_go_d,
                 rel_gdp_pw_d)
    print(f'\033[1;033mObjetivo: {np.around(fobj, 8)}, iter: {cc}') 


#%%


## Produtividade não altera consumo, altera o preço
## ao usar o consumo multiplicado pelo preço america como target
## as mudanças na tfp não alteram o consumo final
## talvez funciona se dividir pelo labor


Bd = ( (0.01, 100), )*i  
my_iter = 5000

cc=0
##  Nelder-Mead
sol = minimize(obj1,
               x1.flatten(),
               method='Nelder-Mead', 
               bounds = Bd, 
               args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d, share_go_d, rel_gdp_pw_d),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter})


out(A, share_lab_d, share_go_d, rel_gdp_pw_d)




#%%

Bd = ( (0.01, 10), )*i  


sol = differential_evolution(
    obj1,
    bounds=Bd,
    args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d, share_go_d, rel_gdp_pw_d),
    strategy='randtobest1bin',      
    tol=1e-10,                 
    mutation=(0.5, 1),         
    recombination = 0.9,        
    maxiter=500,              
    popsize=25,                
    polish=True,               
    disp=True,                 
    updating='deferred',       
)


##  Nelder-Mead
sol = minimize(obj1,
               sol.x,
               method='Nelder-Mead', 
               bounds = Bd, 
               args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d, share_go_d, rel_gdp_pw_d),
               callback = callback, 
               tol = 1e-8,
               options={'maxiter': my_iter})


out(sol.x, share_lab_d, share_go_d, rel_gdp_pw_d)


#%%

#### RUN FOR ALL COUNTRIES ####

#Bd = ( (0.001, 500), )*i  
#Bd = ( (0.001, 1000), )*i  

res = []
Bd = ( (0.001, 5), )*i  
my_iter = 5000


#
for country in dt.ccode:
    
    print('\033[1;033m---'*11)       
    print(f'Country: {country}')
    
    # taus
    tau_w = dt.tau_w_f(code = country, ano = year)
    tau_j = dt.tau_j_f(code = country, ano = year)
      
    #  labor share for a specific year
    share_lab_d = dt.compute_dataset(code= country, ano = year)[0]
    
    # share of gross output for a specific year
    share_go_d = dt.compute_dataset(code = country, ano = year)[2]

    
    ## GDP per worker relative to US for a specific year ##
    
    # GDP per worker - Other countries
    gdp_tot = np.sum(np.sum(dt.fim_use_f(code = country, year = year), axis = 1) )
    emp_tot = np.sum(np.array(dt.sea_f('EMP', code = country, ano = 2014)))
    
    gdp_pw = gdp_tot/emp_tot
    
    ## relative GDP per worker
    rel_gdp_pw_d = gdp_pw/gdp_pw_usa

    
    # total labor
    L = 1#np.sum(np.array(dt.sea_f(var = 'EMP', code = country, ano = year) ) ) / 1e5 # change scale
    
    print('\033[1;033mRun Differential Evolution!')
    ## Run solver
    sol = differential_evolution(
            obj1,
            bounds=Bd,
            args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d, share_go_d, rel_gdp_pw_d),
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
    
    ## force agriculture's tfp equal one
    sol.x[0] = 1
    
    ##  Nelder-Mead
    sol = minimize(
            obj1,
            sol.x,
            method='Nelder-Mead', 
            bounds = Bd, 
            args= (c_bar, alpha, beta, sigma, L, tau_j, tau_w, share_lab_d, share_go_d, rel_gdp_pw_d),
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
    
    ## share go model
    share_go_m = np.multiply(model[3], model[7]) #
    share_go_m = (share_go_m/np.sum(share_go_m))
    
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
            'share_go_m': share_go_m[ii].item(),
            'share_go_d': share_go_d[ii].item(),
            'share_lab_m': model[8][ii].item(),
            'share_lab_d': share_lab_d[ii].item(),
            'tfp': sol.x[ii].item(),       
            'GDP': tot_lab, 
            'obj': sol.fun,
            'success': sol.success
            }
        )





df = pd.DataFrame(res)

df.to_excel('results_go.xlsx', index = False)



















