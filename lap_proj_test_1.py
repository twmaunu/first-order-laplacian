import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import utils

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) 

# set parameters
n = 20
k = 2
p = 0.9
q = 0.1
T = n
niter = 10000
opt = 'densesbm'

# create random Laplacian matrix
L = utils.create_graph_lap(n, k, p, q, opt)
L = L / np.trace(L) * T
N = np.random.randn(n,n)*0.01
N = (N + N.T)/2
Ln = L + N

Lhat1, convfw, timesfw = utils.lap_proj(Ln, T, niter, step = 1, opt = 'fw', ls = False)
Lhat2, convmd, timesmd = utils.lap_proj(Ln, T, niter, step = 10, opt = 'md')

plt.figure()
plt.plot(timesfw, np.log10(convfw), label = 'FW')
plt.plot(timesmd, np.log10(convmd), label = 'MD')
plt.ylabel(r'$\log_{10}(\|L_* - \hat{L}\|_{F}^{2})$')
plt.xlabel('Time (s)')
plt.legend()
plt.savefig('lap_proj_test_1.png', dpi = 300)
plt.grid()
plt.show()

# plt.figure()
# plt.imshow(L)
# plt.colorbar()
# plt.show()