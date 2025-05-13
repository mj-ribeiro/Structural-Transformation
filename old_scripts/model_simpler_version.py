# 
# ################################################################################################
# 
# #                 Structural Transformation and Network Effects
# #                            R&R Economic Modelling
# 
# ################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 29/03/2025
# ################################################################################################
# #### In this script I made the functions of model 
# I made a simple version with two sectors only to test 


#%%

import os
os.getcwd()
os.chdir(r"C:\Users\marco\Downloads\paper_EM")

import numpy as np
import pandas as pd
import get_data as dt
from scipy.optimize import minimize

np.set_printoptions(suppress=True)
np.set_printoptions(precision = 4)

# function to format output print of a vectors
num = lambda x: np.array2string(np.squeeze(x), formatter={'float_kind': lambda x: f"{x:.3f}"}, separator=", ").strip("[]")


# %%
#### Define A guess
def tfp(i):    
    A = np.ones(i).reshape(i, 1)
    return A


## Guess alpha and c_bar
def x1_f(i):
    alpha = np.array([0.1, 0.9]).reshape(i, 1)
    c_bar = np.array([0.003, 0.001]).reshape(i, 1)
    return np.concatenate([alpha.flatten(), c_bar.flatten()])

## Parameters
i = 2
A = tfp(i)
beta = np.array([0.1, 0.9, 0.2, 0.8]).reshape(i,i) #dt.beta_f(code = 'USA', ano = 2000)
sigma = np.array([0.3, 0.9]).reshape(i,1) #dt.sigma_f(code = 'USA', ano = 2000)
w = np.ones(i).reshape(i, 1)
x1 = x1_f(i)

## "dataset"
share_lab_d = np.array([0.2, 0.8]).reshape(i, 1)
share_cons_d = np.array([0.55, 0.45]).reshape(i, 1)

beta = np.array([0.3, 0.7, 0.44, 0.56]).reshape(i,i) 

#%%
############# Compute Equilibrium 
def equilibrium(x1, beta, sigma, L, i = 2):
    
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
    ## Calcular L_i sem ajuste
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
     share_lab_m] = equilibrium(x1 = x1, beta = beta, sigma = sigma, L = 100)
    
    obj = np.sum( np.power(np.divide((share_cons_m - share_cons_d), share_cons_m), 2) ) +   \
    np.sum( np.power(np.divide(( share_lab_m -  share_lab_d ),  share_lab_m ), 2) )
    
    #prices: 3.35, 5.99
    print(f'\033[1;33m Objective Function: {num(obj)}')
    print(f'Sectorial Labor: {num(Li)}')
    print(f'Total Labor: {num(np.sum(Li)) }')
    print(f'Labor share: {num(share_lab_m)}')
    print(f'GDP sec: {num(GDP_sec)}')
    print(f'GDP: {num(GDP)}')
    print(f'Prices: {num(p)}')
    print(f'Share of consumption: {num(share_cons_m)}')    
    # restrição tem que dar zero
    print(f'Constraint: {num(np.multiply(G, Li) - np.sum(np.multiply(B, Li), axis = 0).reshape(i, 1) - C)}')

# sigma pequenos aumentam o p, isso distorce os resultados
# mudanças em c_bar afetam a qtd de trabalhadores, mas não o share de trabalhadores
#%%
alpha = np.array([0.7, 0.3]).reshape(i, 1)
c_bar = np.array([0.25, 0.2]).reshape(i, 1)#np.array([0.003, 0.001]).reshape(i, 1)
x1 = np.concatenate([alpha.flatten(), c_bar.flatten()])
sigma = np.array([0.1, 0.2]).reshape(i,1) #dt.sigma_f(code = 'USA', ano = 2000)


out(x1)


#%%

cc = 0
def callback(x1):    
    global cc
    cc += 1
    fobj = obj1(x1)
    print(f'\033[1;033mObjetivo: {np.around(fobj, 8)}, iter: {cc}') 


def obj1(x1):
    
    x1[:i] = x1[:i]/np.sum(x1[:i])
    
    [p,
     C,     
     B, 
     G,
     GDP,
     GDP_sec,
     share_cons_m,
     labor_sec_m,
     share_lab_m] =  equilibrium(x1 = x1,
                                 beta = beta, 
                                 sigma = sigma, L = 100)                   
    
    obj = np.sum( np.power(np.divide((share_cons_m - share_cons_d), share_cons_m), 2) ) +  \
             np.sum( np.power(np.divide(( share_lab_m -  share_lab_d ),  share_lab_m ), 2) )

    #obj = np.log(obj)
    return obj


#%%
sigma = np.array([0.6, 0.5]).reshape(i,1) #dt.sigma_f(code = 'USA', ano = 2000)
alpha = np.array([0.4, 0.6]).reshape(i, 1)
c_bar = np.array([25, 2.12]).reshape(i, 1)#np.array([0.003, 0.001]).reshape(i, 1)
x1 = np.concatenate([alpha.flatten(), c_bar.flatten()])

Bd = ((1e-5, 0.999), )*i + ( (0.01, 2), )*i  

my_iter = 1e4
cc=0
sol = minimize(obj1,
               x1,
               method='Nelder-Mead', 
               bounds = Bd, 
               callback = callback, 
               tol = 1e-20,
               options={'maxiter':my_iter,
                        'maxls':800, 
                        'maxfun':1e10,  
                        'maxcor': 3000, 
                        'eps': 1e-08})

#%%


x1_test = np.array([0.5, 0.5, 0.01, 0.01])  # Um ponto intermediário
sol = minimize(obj1, x1_test, method='L-BFGS-B', bounds=Bd, tol=1e-10)
sol


#%%

[p,
 C,
 B, 
 G,
 GDP,
 GDP_sec,
 share_cons_m,
 labor_sec_m,
 share_lab_m] =  equilibrium(x1 = sol.x,
                      beta = beta, 
                      sigma = sigma, 
                      L = 100)

print(f'labor share model: {num(share_lab_m)}')
print(f'labor share data: {num(share_lab_d)}')

print(f'Share of consumption model: {num(share_cons_m)}')
print(f'Share of consumption data: {num(share_cons_d)}')

print(f'resultado: {sol.x}')


# In[ ]:




####################################################################### 
# m = np.array([1, 1, 2,0]).reshape(2, 2)
# s = np.array([-1, -1, 1, 1]).reshape(2, 2)   

# # matrix product
# np.diag(m@s.T)       

# # element wise product 
# np.sum(m*s, axis = 1)
# ### Equivalences between matrix operations
# ###  Matrices Nx1 and Nx1

# ## Nx1 vs Nx1 without somation  -> G_i L_i
# np.multiply(G, Li)

# np.diag(G@Li.T).reshape(i, 1) # divide both sides by G


# ### Matrices NxN vs Nx1 -> \sum_i B_{ij} L_i
# np.sum(np.multiply(B, Li), axis = 0)

# B.T@Li    



# ## Matrices Nx1 and Nx1 multiplied by a vetor  -> alpha/p Li

# alpha/p* np.sum(Li, axis = 0)

# np.eye(i)@(alpha/p)@np.ones(i).reshape(1, i)@Li


# # matrix product  - prices
# D = np.eye(4)*sig1

# np.multiply(sig1, np.sum(np.multiply(beta, p), axis = 0).T)    

# D@beta.T@p    

# # ### Tests
# # np.multiply(G, Li) - np.sum(np.multiply(B, Li), axis = 1) - \
# #     np.multiply(alpha/p, np.sum(Li)) - c_bar + np.multiply(alpha/p, np.sum(np.multiply(p, c_bar) ))
# # ############################################

# Li_init = np.ones_like(G)  # Chute inicial

# def equation(Li, G, B, alpha, p, c_bar):

#     term1 = np.multiply(G, Li.reshape(i, 1) )
#     term2 = np.sum(np.multiply(B, Li.reshape(i, 1)), axis = 1)
#     term3 = np.multiply(alpha / p, np.sum(Li.reshape(i, 1)))
#     term4 = c_bar
#     term5 = np.multiply(alpha / p, np.sum(np.multiply(p, c_bar)))

#     return np.array(term1 + term2 + term3 + term4 + term5).reshape(i,)

# def find_Li(G, B, alpha, p, c_bar, Li_init):

#     result = root(equation, Li_init, args=(G, B, alpha, p, c_bar))
#     if result.success:
#         return result.x
#     else:
#         raise ValueError("Não foi possível encontrar uma solução.")

# Lsolve = find_Li(G, B, alpha, p, c_bar, Li_init).reshape(i, 1)

# #equation(Lsolve, G, B, alpha, p, c_bar)  
# #######################################################################




