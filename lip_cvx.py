from math import inf
import numpy as np
import scipy as sp
from scipy.linalg import block_diag
import cvxpy as cp
import sys, time

def zeros(s1, s2):
    return cp.Constant(np.zeros((s1, s2)))

def to_cvx(a):
    # converts numpy array to picos constant
    return cp.Constant(a.astype(np.double))

def blkdiag_old(a, au=None, al=None): # deprecated
    n = len(a)
    for i in range(n):
        if sum(a[i].shape) < 1:
            return blkdiag(a[:i] + a[i+1:])
    if n < 2:
        return a[0]
    if au is None and al is None:
        d0, d1 = [ai.shape[0] for ai in a], [ai.shape[1] for ai in a]
        res = a[0] & zeros(d0[0], sum(d1[1:n]))
        for i in range(1, n-1):
            res = res // (zeros(d0[i], sum(d1[0:i])) & a[i] & zeros(d0[i], sum(d1[i+1:n])))
        res = res // (zeros(d0[-1], sum(d1[:-1])) & a[-1])
        return res
    elif len(au) != n-1 or len(al) != n-1:
        raise ValueError
    else:
        # block diagonal matrix with upper and lower secondary diagonal
        d0, d1 = [ai.shape[0] for ai in a], [ai.shape[1] for ai in a]
        res = a[0] & au[0] & zeros(d0[0], sum(d1[2:n]))
        for i in range(1, len(a)-1):
            if i-1 > 0:
                z1 = zeros(d0[i], sum(d1[0:i-1]))
                if i+2 < n:
                    z2 = zeros(d0[i], sum(d1[i+2:n]))
                    res = res // (z1 & al[i-1] & a[i] & au[i] & z2)
                else:
                    res = res // (z1 & al[i-1] & a[i] & au[i])
            else:
                if i+2 < n:
                    z2 = zeros(d0[i], sum(d1[i+2:n]))
                    res = res // (al[i-1] & a[i] & au[i] & z2)
                else:
                    res = res // (al[i-1] & a[i] & au[i])

        res = res // (zeros(d0[-1], sum(d1[:-2])) & al[-1] & a[-1])
        return res

def blkdiag(a, au=None, al=None):
    # creates cvxpy block diagonal, optionally with upper and lower block diagonals
    # both au and al need to be valid lists of expressions or both need to be None
    n = len(a)
    for i in range(n):
        if sum(a[i].shape) < 1:
            return blkdiag(a[:i] + a[i+1:])
    if n < 2:
        return a[0]
    
    d0, d1 = [ai.shape[0] for ai in a], [ai.shape[1] for ai in a]
    if au is None and al is None:
        f = [a[0], zeros(d0[0], sum(d1[1:n]))]
        m = [[zeros(d0[i], sum(d1[0:i])), a[i], zeros(d0[i], sum(d1[i+1:n]))] for i in range(1, n-1)]
        l = [zeros(d0[n-1], sum(d1[0:n-1])), a[-1]]
        res = cp.Bmat([f, *m, l])
        return res
    else:
        f1 = [a[0], au[0], zeros(d0[0], sum(d1[2:n]))]
        f2 = [al[0], a[1], au[1], zeros(d0[1], sum(d1[3:n]))]
        m = None
        if n > 5:
            m = [[zeros(d0[i], sum(d1[0:i-1])), al[i-1], a[i], au[i], zeros(d0[i], sum(d1[i+2:n]))] for i in range(1, n-2)]
        l1 = [zeros(d0[n-2], sum(d1[0:n-3])), al[n-3], a[n-2], au[n-2]]
        l2 = [zeros(d0[n-1], sum(d1[0:n-2])), al[n-2], a[n-1]]
        if m is None:
            res = cp.bmat([f1, f2, l1, l2])
        else:
            res = cp.bmat([f1, f2, *m, l1, l2])
        return res

def solveLipSDP(weights, return_Q=False):
    dims = [weight.shape[1] for weight in weights]
    dims.append(weights[-1].shape[0])
    n = len(dims) - 1
    L2 = cp.Variable(nonneg=True)
    sdims = sum(dims)
    maxdimsum = inf # adjust if approximation needed for big nets
    feasible = sdims < maxdimsum
    if feasible:
        lamdim = sum(dims[1:])
        lambdas = cp.Variable((lamdim,1), nonneg=True)
    else:
        print("USING APPROXIMATION! (NET SPLIT IN MIDDLE, ONE SCALAR LAMBDA)")
        lambdas = cp.Variable((1,1), nonneg=True)
        
        # split net in middle of layers
        mi = int((n+1)/2)
        weights1 = weights[:mi]
        weights2 = weights[mi:]
        # split net at layer in middle of neuron count
        for i in range(1,len(dims)-1):
            if sum(dims[:i]) > sdims/2:
                weights1 = weights[:i]
                weights2 = weights[i:]
                break

        # estimate lipschitz constants for split nets, return their product
        l1, lambdas1 = solveLipSDP(weights1)
        l2, lambdas2 = solveLipSDP(weights2)

        return l1*l2, [lambdas1, lambdas2]

    ldiags = None
    if feasible:
        ldiags = [cp.diag(lambdas[sum(dims[1:i+1]):sum(dims[1:i+2])]) for i in range(n-1)]
    else:
        ldiags = [lambdas * to_cvx(np.eye(dims[i+1])) for i in range(n-1)]

    l2m = L2*np.eye(dims[0])
    I_end = to_cvx(np.eye(dims[-1]))

    a = [l2m] + [2*ldiag for ldiag in ldiags] + [I_end] # main diagonal
    au = [-to_cvx(np.transpose(weights[i])) @ ldiags[i] for i in range(n-1)] + [-to_cvx(np.transpose(weights[n-1]))] # upper diagonal
    al = [-ldiags[i] @ to_cvx(weights[i]) for i in range(n-1)] + [-to_cvx(weights[n-1])] # lower diagonal

    Q = blkdiag(a, au=au, al=al)
    # Q is calculated

    eps = 1e-6

    # define problem
    obj = cp.Minimize(L2)
    constraints = [Q >> eps*1]

    p = cp.Problem(obj, constraints)
    p.solve(solver='CVXOPT', verbose=False)

    # with np.printoptions(linewidth=np.inf):
    #     print(Q.value)

    if return_Q:
        return Q.value
    else:
        return np.sqrt(L2.value), lambdas.value

if __name__ == "__main__":
    # print(pc.solvers.available_solvers())
    # sys.exit()
    weights = np.load("weights.npy", allow_pickle=True)
    time1 = time.perf_counter()
    l, lambdas = solveLipSDP(weights)
    diff = time.perf_counter() - time1
    print("Calculation finished after ", round(diff, 2), "s")
    print("Lipschitz constant upper bound:", round(l, 7))

