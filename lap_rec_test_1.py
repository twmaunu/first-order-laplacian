import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import utils




# set parameters
ns = [10, 20, 30]
k = 2
p = 0.95
q = 0.05
T = 1
niter = 400
reps = 5
opt = 'sbm'

convfw = np.zeros((len(ns), reps, niter))
convmd = np.zeros((len(ns), reps, niter))
timesfw = np.zeros((len(ns), reps, niter))
timesmd = np.zeros((len(ns), reps, niter))



for i in range(len(ns)):
    for j in range(reps):
        n = ns[i]
        m = 2 * n ** 2

        # create random Laplacian matrix
        L = utils.create_graph_lap(n, k, p, q, opt)
        L = L / np.trace(L) * T

        X = np.random.normal(0,1,(m, n))
        Y = np.sum((X @ L) * X, axis = 1)

        _, cfw, tfw = utils.lap_recover(X, Y, T, L, niter, opt = 'fw', step = 1)
        _, cmd, tmd = utils.lap_recover(X, Y, T, L, niter, opt = 'md', step = 10)
        convfw[i, j, :] = cfw
        convmd[i, j, :] = cmd
        timesfw[i, j, :] = tfw
        timesmd[i, j, :] = tmd


print(convfw.shape)

plt.figure()
colorsred = plt.cm.Reds(np.linspace(0.5, .9, len(ns)))[::-1]
colorsblue = plt.cm.Blues(np.linspace(0.5, .9, len(ns)))[::-1]
for i in range(len(ns)):
    plt.plot(np.log10(np.mean(convmd[i], axis = 0)).T, c = colorsblue[i])
    plt.plot(np.log10(np.mean(convfw[i], axis = 0)).T, c = colorsred[i])
# plt.plot(timesmd.T, np.log10(convmd).T, 'r', cmap = 'Blues')
# plt.plot(timesfw.T, np.log10(convfw).T, 'b', cmap = 'Reds')
plt.legend(['MD', 'FW'])
plt.ylabel('$\log_{10}(\|L_* - L_{hat}\|_{F}^{2})$', interpreter = 'latex')
plt.grid()
plt.savefig('lap_rec_test_1.png', dpi = 300)
plt.show()