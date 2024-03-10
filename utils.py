import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import time
from tqdm import tqdm

# create random Laplacian matrix
def create_graph_lap(n, k, p, q, opt = 'sbm'):
    """ Create a random Laplacian matrix"""
    if opt == 'sbm':
        nk = int(n/k)
        A = np.zeros((n,n))
        for i in range(k):
            for j in range(i+1):
                if i == j:
                    B = np.random.rand(nk,nk)
                    B = (B < p)
                    A[i*nk:(i+1)*nk, j*nk:(j+1)*nk] = np.tril(B,-1)
                else:
                    B = np.random.rand(nk,nk)
                    B = (B < q)
                    A[i*nk:(i+1)*nk, j*nk:(j+1)*nk] = B.copy()
                
        L = np.tril(A,-1)

        L = L + L.T
        L = np.diag(L @ np.ones(n)) - L

    elif opt == 'erdos':
        A = np.random.rand(n,n)
        A = (A < p)
        L = np.tril(A,-1)
        L = L + L.T
        L = np.diag(L @ np.ones(n)) - L

    elif opt == 'densesbm':
        nk = int(n/k)
        A = np.zeros((n,n))
        for i in range(k):
            for j in range(i+1):
                if i == j:
                    B = 1-(np.random.rand(nk,nk)*(1-p))
                    
                    A[i*nk:(i+1)*nk, j*nk:(j+1)*nk] = np.tril(B,-1)
                else:
                    B = np.random.rand(nk,nk)*q + q
                    A[i*nk:(i+1)*nk, j*nk:(j+1)*nk] = B.copy()
        L = np.tril(A,-1)

        L = L + L.T
        L = np.diag(L @ np.ones(n)) - L

    return L

def lap_fw(n, T, F, gradF, niter = 1000, step = 1, opt = 'A', ls = True):
    
    Li = np.eye(n) - np.ones((n,n))/n+np.eye(n)/n
    Li = T * Li / np.trace(Li)  
    Ai = np.diag(np.diag(Li)) - Li

    conv = np.zeros(niter)
    times = np.zeros(niter)
    t0 = time.time()
    for i in tqdm(range(niter)):
        times[i] = time.time()-t0
        if opt == 'A':
            conv[i] = F(Ai)
            # compute gradient
            grad = gradF(Ai)
            # find FW step direction
            tmp = grad + np.diag(np.inf * np.ones(n))
            (j1,j2) = np.unravel_index(np.argmin(tmp), shape = (n,n))
            #print(j1,j2)
            G = np.zeros((n,n))
            G[j1,j2] = 1
            G[j2,j1] = 1
            G = G / 2 * T

            # compute step size
            if ls:
                scale = 1/2
                eta = 1/2
                for i in range(12):
                    if F(Ai + (eta - 0.5*scale)*(G-Ai) (G-Ai)) < F(Ai + (eta + 0.5*scale) * (G-Ai)):
                        eta = eta - 0.5 * scale
                        scale = scale / 2
                    else:
                        eta = eta + 0.5 * scale
                        scale = scale / 2
            else:
                eta = step * 2/(i+3)
            # update
            Ai = (1-eta) * Ai + eta * G
        if opt == 'L':
            conv[i] = F(Li)
            # compute gradient
            grad = gradF(Li)

            grad = -grad
            tmp = grad + np.diag(np.inf * np.ones(n))
            (j1,j2) = np.unravel_index(np.argmin(tmp), shape = (n,n))
            #print(j1,j2)
            G = np.zeros((n,n))
            G[j1,j2] = -1
            G[j2,j1] = -1
            G[j1, j1] = 1
            G[j2, j2] = 1
            G = G / 2 * T

            # compute step size
            if ls:
                scale = 1/2
                eta = 1/2
                for i in range(12):
                    if F(Li + eta * (G-Li) - 0.5*scale*(G-Li)) < F(Li + eta * (G-Li) + 0.5*scale*(G-Li)):
                        eta = eta - 0.5 * scale
                        scale = scale / 2
                    else:
                        eta = eta + 0.5 * scale
                        scale = scale / 2
            else:
                eta = step * 2/(i+3)

            Li = (1-eta) * Li + eta * G
            Ai = np.diag(np.diag(Li)) - Li

    Li = np.diag(Ai @ np.ones(n)) - Ai
    return Li, Ai, conv, times

def lap_md(n, T, F, gradF, niter = 1000, step = 1, opt = 'A'):

    Li = np.eye(n) - np.ones((n,n))/n+np.eye(n)/n
    Li = T * Li / np.trace(Li)  
    Ai = np.diag(np.diag(Li)) - Li

    conv = np.zeros(niter)
    times = np.zeros(niter)
    t0 = time.time()
    for i in tqdm(range(niter)):
        times[i] = time.time()-t0
        if opt == 'A':
            conv[i] = F(Ai)
            # compute gradient
            grad = gradF(Ai)
            

            Ai = Ai * np.exp(-step * grad/np.sqrt(1))
            Ai = Ai / np.sum(Ai) * T
        if opt == 'L':
            conv[i] = F(Li)
            # compute gradient
            grad = gradF(Li)
            grad = grad

            Ai = np.log(Ai)+step * grad/np.sqrt(1)
            Ai = np.exp(Ai)
            Ai = Ai / np.sum(Ai) * T
            Li = np.diag(Ai @ np.ones(n)) - Ai

    Li = np.diag(Ai @ np.ones(n)) - Ai
    return Li, Ai, conv, times

# function to project Laplacian matrix
def lap_proj(S, T, niter = 1000, step = 1, opt = 'fw', ls = True):
    """ Project a matrix to a graph laplcian matrix"""
    #niter = 1000
    conv = np.zeros(niter)

    n = S.shape[0]

    Li = np.eye(n) - np.ones((n,n))/n+np.eye(n)/n
    Li = T * Li / np.trace(Li)  
    Ai = np.diag(np.diag(Li)) - Li
    AS = np.diag(np.diag(S)) - S
    conv_fw = np.zeros(niter)

    F = lambda A: np.linalg.norm(A - AS)**2
    gradF = lambda A: (A - AS) + ((A - AS) @ np.ones(AS.shape[0])).reshape(AS.shape[0],1) + ((A - AS) @ np.ones(AS.shape[0])).reshape(1,AS.shape[0])
    
    FL = lambda A: np.linalg.norm(A - S)**2
    gradFL = lambda A: (A - S)
    

    if opt == 'fw':
        L, A, conv, times = lap_fw(n, T, F, gradF, step = 1, niter = niter, opt = 'A', ls = ls)
        # L, A, conv, times = lap_fw(n, T, FL, gradFL, niter = niter,  step = step, opt = 'L', ls = ls)
         
    if opt == 'md':
         L, A, conv, times = lap_md(n, T, F, gradF, niter = niter, opt = 'A')
        #  L, A, conv, times = lap_md(n, T, FL, gradFL, step = step, niter = niter, opt = 'L')
       
    return L, conv, times




# function to recover Laplacian matrix from quadratic measurments
def lap_recover(X, Y, T, Lstar, niter = 1000, opt = 'fw', step = 1, copt = 'LS', ls = True):
    
    [m,n] = X.shape
    Cx = X.T @ X / m

    Cx2 = sp.linalg.sqrtm(Cx)
    Cx2i = np.linalg.inv(Cx2)
    X2 = X @ np.linalg.inv(Cx2)
    if copt == 'LS':
        F = lambda A: np.linalg.norm((np.sum((X @ A) * X, axis = 1) - Y))**2
        gradF = lambda A: X.T @ (np.outer((np.sum((X @ A) * X, axis = 1) - Y), np.ones(A.shape[0])) * X) / X.shape[0]
    if copt == 'BW':
        F = lambda A: np.linalg.norm((np.sqrt(np.sum((X @ A) * X, 1)) - np.sqrt(Y)))**2
        gradF = lambda A: X.T @ (np.outer((np.sqrt(np.sum((X @ A) * X, 1)) - np.sqrt(Y))/np.sqrt(np.sum((X @ A) * X, 1)), np.ones(n)) * X)



    if opt == 'fw':
        L, A, conv, times = lap_fw(n, T, F, gradF, niter = niter, step = step, opt = 'L', ls = ls)
    if opt == 'md':
        L, A, conv, times = lap_md(n, T, F, gradF, niter = niter, step = step, opt = 'L')

    return L, conv, times


# function to recover Laplacian matrix from quadratic measurments
def lap_recover_reg(X, Y, T, Lstar, niter = 1000, opt = 'fw', step = 1, copt = 'LS', ls = True, lamb = 1):
    
    [m,n] = X.shape
    Cx = X.T @ X / m

    Cx2 = sp.linalg.sqrtm(Cx)
    Cx2i = np.linalg.inv(Cx2)
    X2 = X @ np.linalg.inv(Cx2)

    F = lambda A: np.linalg.norm((np.sum((X @ A) * X, axis = 1) - Y))**2 / (2*X.shape[0]) - lamb * np.trace(A) 
    gradF = lambda A: X.T @ (np.outer((np.sum((X @ A) * X, axis = 1) - Y), np.ones(A.shape[0])) * X) / X.shape[0] - lamb * np.eye(A.shape[0])



    if opt == 'fw':
        L, A, conv, times = lap_fw(n, T, F, gradF, niter = niter, step = step, opt = 'L', ls = ls)
    if opt == 'md':
        L, A, conv, times = lap_md(n, T, F, gradF, niter = niter, step = step, opt = 'L')

    return L, conv, times
    