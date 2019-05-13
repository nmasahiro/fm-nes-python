#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import function
import random, pickle, json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime, os
from utils import load_class
import pandas as pd
import sys, math, time
from scipy.linalg import expm
from functools import reduce
import functools

class RealSolution(object):
    def __init__(self, **params):
        self.f = float('nan')
        self.x = np.zeros([params['dim'], 1])
        self.z = np.zeros([params['dim'], 1])

def get_h_inv(dim):
    f = lambda a,b: ((1. + a*a)*math.exp(a*a/2.) / 0.24) - 10. - dim
    fprime = lambda a: (1. / 0.24) * a * math.exp(a*a/2.) * (3. + a*a)
    h_inv = 1.0
    while (abs(f(h_inv, dim)) > 1e-10):
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / fprime(h_inv))
    return h_inv

def comparator(a, b):
    sgn = 0
    if a.f < sys.maxsize and b.f < sys.maxsize:
        if a.f - b.f > 0:
            sgn = 1
        elif a.f - b.f < 0:
            sgn = -1
        return sgn
    elif a.f < sys.maxsize and b.f >= sys.maxsize:
        return -1
    elif a.f >= sys.maxsize and b.f < sys.maxsize:
        return 1
    elif a.f >= sys.maxsize and b.f >= sys.maxsize:
        if np.linalg.norm(a.z) - np.linalg.norm(b.z) > 0:
            sgn = 1
        elif np.linalg.norm(a.z) - np.linalg.norm(b.z) < 0:
            sgn = -1
        return sgn


def main(path, **params):
    obj_func = params['obj_func']
    dim   = params['dim']
    lamb  = params['lamb']
    m     = params['mean']
    sigma = params['sigma']
    B = params['B']
    eta_m = params['eta_m']
    weights_rank_hat = np.array([max(0., math.log(lamb/2. + 1.) - math.log(i+1)) for i in range(lamb)])
    weights_rank = weights_rank_hat / sum(weights_rank_hat) - 1./lamb
    mueff = 1 / np.sum((weights_rank + 1./lamb)**2, axis=0)
    cs = (mueff + 2.) / (dim + mueff + 5.)
    cc = (4.+mueff/dim) / (dim+4.+2.*mueff/dim)
    c1_cma = 2. / (math.pow(dim+1.3,2) + mueff)
    # initialization
    chiN = np.sqrt(dim)*(1.-1./(4.*dim)+1./(21.*dim*dim))
    pc = np.zeros([dim, 1])
    ps = np.zeros([dim, 1])
    # distance weight parameter
    h_inv = get_h_inv(dim)
    alpha_dist = lambda lambF: h_inv * min(1., math.sqrt(float(lamb)/dim)) * math.sqrt(float(lambF)/lamb)
    w_dist_hat = lambda z, lambF: math.exp(alpha_dist(lambF) * np.linalg.norm(z))
    # learning rate
    eta_move_sigma = 1.
    eta_stag_sigma = lambda lambF: math.tanh((0.024*lambF + 0.7*dim + 20.)/(dim + 12.))
    eta_conv_sigma = lambda lambF: 2. * math.tanh((0.025*lambF + 0.75*dim + 10.)/(dim + 4.))
    c1 = lambda lambF: 2. / (math.pow(dim + 1.3, 2.) + mueff) * (float(lambF)/lamb)
    eta_B = lambda lambF: 120.*dim / (47.*dim*dim + 6400.) * math.tanh(0.02 * lambF)
    # eiadx-nes
    cGamma = 1. / (3.*(dim-1.))
    dGamma = min(1., float(dim)/lamb)
    gamma = 1.

    no_of_evals = 0
    g = 0

    solutions = [RealSolution(**params) for i in range(lamb)]
    while no_of_evals < params['max_evals']:
        g += 1

        for i in range(int(lamb/2)):
            solutions[2*i].z = np.random.randn(dim, 1)
            solutions[2*i+1].z = -solutions[2*i].z.copy()
            solutions[2*i].x = m + sigma * B.dot(solutions[2*i].z)
            solutions[2*i+1].x = m + sigma * B.dot(solutions[2*i+1].z)
            solutions[2*i].f = obj_func.evaluate(solutions[2*i].x)
            solutions[2*i+1].f = obj_func.evaluate(solutions[2*i+1].x)
        no_of_evals += lamb

        solutions = sorted(solutions, key=functools.cmp_to_key(comparator)) 

        best = solutions[0].f
        if g % 100 == 0:
            print("best:", best)

        if solutions[0].f < params['criterion']:
            # print(no_of_evals, solutions[0].f)
            break

        lambF = len([solutions[i].x for i in range(lamb) if solutions[i].f < sys.maxsize])

        # evolution path p_sigma
        wz = np.sum([weights_rank[i]*solutions[i].z for i in range(lamb)], axis=0)
        ps = (1.-cs) * ps + np.sqrt(cs * (2.-cs) * mueff) * wz
        ps_norm = np.linalg.norm(ps)

        # distance weight
        w = [weights_rank_hat[i] * w_dist_hat(solutions[i].z, lambF) for i in range(lamb)]
        weights_dist = w / sum(w) - 1./lamb

        # switching weights and learning rate
        weights = weights_dist if ps_norm >= chiN else weights_rank
        eta_sigma = eta_move_sigma if ps_norm >= chiN else eta_stag_sigma(lambF) if ps_norm >= 0.1*chiN else eta_conv_sigma(lambF)

        # calculate natural gradient
        G_delta = reduce(lambda a,b: a+b, [weights[i] * solutions[i].z for i in range(lamb)])
        G_M = reduce(lambda a,b: a+b, [weights[i]*(np.dot(solutions[i].z, solutions[i].z.T) - np.eye(dim, dtype=float)) for i in range(lamb)])
        G_sigma = G_M.trace() / dim
        G_B = G_M - G_sigma * np.eye(dim)

        m += eta_m * sigma * np.dot(B, G_delta)
        pc = (1.-cc) * pc + np.sqrt(cc * (2.-cc) * mueff) * B.dot(G_delta)

        lsig = 1.0 if G_sigma < 0. and ps_norm >= chiN else 0.0
        sigma *= math.exp((1.-lsig) * (eta_sigma/2.) * G_sigma)

        A = np.dot(B.dot(expm(eta_B(lambF) * G_B)), B.T)

        # eiadx-nes
        BBt = B.dot(B.T)
        e, v = np.linalg.eigh(BBt)
        tau_vec = [np.dot(np.dot(v[:,i].reshape(1,dim), A), v[:,i].reshape(dim,1))/np.dot(np.dot(v[:,i].reshape(1,dim), BBt), v[:,i].reshape(dim,1)) - 1. for i in range(dim)]
        tau_flag = [1. if tau_vec[i] > 0 else 0. for i in range(dim)]
        tau = max(tau_vec)
        gamma = max((1.-cGamma)*gamma + cGamma*math.sqrt(1. + dGamma*tau), 1.)

        Q = (gamma-1.) * reduce(lambda a,b: a+b, [tau_flag[i]*np.dot(v[:,i].reshape(dim,1), v[:,i].reshape(1,dim)) for i in range(dim)]) + np.eye(dim, dtype=float)
        stepsizeQ = math.pow(np.linalg.det(Q), 1./dim)
        sigma *= stepsizeQ
        A = np.dot(np.dot(Q, A), Q) / math.pow(stepsizeQ, 2.)

        lc = 1. if ps_norm >= chiN else 0.
        A += lc * c1(lambF) * (pc.dot(pc.T) - BBt)
        if not (math.isnan(np.linalg.det(A)) or np.linalg.det(A) <= 0):
            A /= math.pow(np.linalg.det(A), 1./dim)
        else:
            print("determinant error;")
            print("det:", np.linalg.det(A))
            sys.exit(0)

        # eigenvalue decomposition
        e, v = np.linalg.eigh(A)
        B = np.dot(v.dot(np.diag(list(map(lambda a: math.sqrt(a), e)))), v.T)

    print(no_of_evals, solutions[0].f, params['seed'])

if __name__ == '__main__':
    params = {}
    #params['seed'] = random.randint(0, 2**32 - 1)
    params['seed'] = 4046532
    np.random.seed(params['seed'])
    params['dim'] = 40
    params['lamb'] = 16
    params['max_evals'] = int(5 * params['dim'] * 1e5)
    params['criterion'] = 1e-12
    params['obj_func_name'] = 'function.EllipsoidFunction'
    # params['obj_func_name'] = 'function.RosenbrockChainFunction'
    # params['obj_func_name'] = 'function.RastriginFunction'
    # params['obj_func_name'] = 'function.ConstraintSphereFunction'
    func = load_class(params['obj_func_name'])
    obj_func = func(params['dim'])
    path = 'log/' + obj_func.name + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    if not os.path.isdir(path):
        os.makedirs(path)
    print('create directory which is ' + path)
    f = open('%s/params_setting.json' % path, 'w')
    json.dump(params, f)
    f.close()

    # params['mean'] = np.zeros([params['dim'], 1])
    params['mean'] = 3. * np.ones([params['dim'], 1])
    params['sigma'] = 2.0
    params['B'] = np.eye(params['dim'], dtype=float)
    params['obj_func'] = obj_func
    params['eta_m'] = 1.0

    start = time.time()
    main(path, **params)
    print("calculate time:", time.time() - start, "sec")
