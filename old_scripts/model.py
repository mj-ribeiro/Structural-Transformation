
################################################################################################

#                 Structural Transformation and Network Effects
#                            R&R Economic Modelling

################################################################################################
# Marcos J Ribeiro
# Última atualização 25/03/2025
################################################################################################
# In this script I made the functions of model



import os
os.getcwd()
os.chdir(r"C:\Users\marco\Downloads\paper_EM")

import numpy as np
import pandas as pd
import get_data as dt
from scipy.optimize import minimize

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
np.set_printoptions(suppress=True)



dt.sea_f(var = 'EMP', code = 'USA', ano = 2000)



#%%
#### Define A guess

def tfp(i):    
    A = np.ones(i).reshape(i, 1)
    return A


#%%

## Guess alpha and c_bar


def x1_f(i):
    alpha = np.array([0.1, 0.3, 0.4, 0.2]).reshape(i, 1)
    c_bar = np.array([0.1, 0.05, 0.5, 1.23]).reshape(i, 1)
    return np.concatenate([alpha.flatten(), c_bar.flatten()])



#%%

## Parameters
i = 4
A = tfp(i)
beta = dt.beta_f(code = 'USA', ano = 2000)
sigma = dt.sigma_f(code = 'USA', ano = 2000) #+ 0.2
i = 4
theta = 3 
w = np.ones(i).reshape(i, 1)
x1 = x1_f(i)


### O segredo é o sigma
## sigmas pequenos fazem o L ficar negativo

#%%
############# Compute Equilibrium 

def equilibrium(x1, beta, sigma, i = 4):
    
    ###### PRICES ########     
    sig1 = 1 - sigma 
    
    ## Matrix Phi
    P1 = - np.log(A) - np.multiply(sig1, np.log(sig1)) - np.multiply(sigma, np.log(sigma)) + \
                                np.multiply(sigma, np.log(w) ) 
    P2 = - np.multiply(sig1, np.sum( np.multiply(beta, np.log(beta) ), axis = 0).reshape(i, 1) )    
    
    PHI = P1 + P2 
    
    ### Matrix D multiplied by matrix sigma
    D = np.multiply(np.eye(i), sig1)
    
    ## Identity matrix
    I = np.eye(i)
    
    # Prices   
    log_p = np.dot(np.linalg.inv(I - np.dot(D, beta.T)), PHI)
    p = np.exp(log_p)
    
    #### Objects B and G ####    
    
    # Object B  (X/L)
    B =  np.multiply(np.multiply(np.divide(sig1, sigma), np.divide(beta, p.T) ), w)   

    # Object G  (Q/L)
    g1 = np.multiply(np.power(np.divide(sig1, sigma), sig1), np.power(w, sig1) )   
    g2 =  np.prod( np.power(np.divide(beta, p.T), np.multiply(beta, sig1) ), axis = 0).reshape(i, 1) 
    G = np.multiply(A, np.multiply(g1, g2))     
    
    ####### LABOUR  #######
    
    #### Auxiliar Matrices #####
    
    ### Define alpha and c_bar
    c_bar = x1[i:].reshape(i, 1); alpha = x1[:i].reshape(i, 1)
    
    # ones vector
    uns = np.ones(i).reshape(i, 1)
    
    # B divide by G
    B_til = np.divide(B, G)
    alpha_til = np.divide(alpha, np.multiply(p, G) )
    
    ## LHS
    M = I - B_til.T - (I@(alpha_til)@uns.T)    
    
    ## RHS
    T = np.divide( (c_bar - np.multiply(alpha/p, np.dot(p.T, c_bar) ) ), G)
    
    #### Labor
    Li = np.array(np.dot(np.linalg.inv(M), T) )
    L = np.sum(Li)    
 
    ## labor share
    share_lab = Li/L
    
    ##### CONSUMPTION ######     
    C = c_bar +  np.multiply( np.divide(alpha, p), (np.multiply(w, L) - np.sum(np.multiply(p, c_bar))) )
  
    #### GDP ####   
    GDP = np.sum(np.multiply(p, C)) 
                                                        
    #### Sectoral GDP ####
    GDP_sec = np.multiply(p, C)
    
    #### share of consumption on GDP
    share_cons = GDP_sec/GDP
        
    return [p, C, B, G, GDP, GDP_sec, share_cons, Li, share_lab]



#%%    


dt.i = 4

#  labor share
share_lab_d = dt.compute_dataset(code= 'USA', ano = 2000)[0]

# share of consumption
share_cons_d = dt.compute_dataset(code = 'USA', ano = 2000)[1]

## guess alpha and c_bar
x1 = x1_f(i)

#%%
obj1(x1)

#%%

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
                          sigma = sigma)
                        
    
    obj = np.sum( np.power(np.divide((share_cons_m - share_cons_d), share_cons_m), 2) ) + \
          np.sum( np.power(np.divide(( share_lab_m -  share_lab_d ),  share_lab_m ), 2) )

    #obj = np.log(obj)
    return obj

#%%
### Callback 

cc = 0
def callback(x1):    
    global cc
    cc += 1
    fobj = obj1(x1)
    print(f'\033[1;033mObjetivo: {np.around(fobj, 4)}, iter: {cc}') 


#%%

Bd = ((1e-5, 0.999), )*i + ( (0.01, 5000), )*i  
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

sol.x

#%%
 

x1_init = x1_f(i)

Bd = ((1e-5, 0.999), )*i + ( (0.0, 50000), )*i  

sol = minimize(obj1, x1_init, method='L-BFGS-B',
               bounds=Bd, 
               callback=callback,
               tol=1e-12,
               options={'maxiter': my_iter})

sol.x



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
                      sigma = sigma)


share_lab_m
share_lab_d

share_cons_d
share_cons_m


np.multiply(p, C)
GDP_sec













## end





