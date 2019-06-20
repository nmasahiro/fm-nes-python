import numpy as np
import sys


def sphere(x):
    return np.sum(x**2)


def ktablet(x):
    n = len(x)
    k = int(n / 4)
    if len(x) < 2:
        raise ValueError('dimension must be greater one')
    return np.sum(x[0:k]**2) + np.sum((100.0*x[k:n])**2)


def one_tablet(x):
    n = len(x)
    return 1e6*(x[0]**2) + np.sum(x[1:n]**2)


def cigar(x):
    n = len(x)
    return x[0]**2 + 1e6*np.sum(x[1:n]**2)


def ellipsoid(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError('dimension must be greater one')
    return np.sum([(1000000**(i / (n-1)) * x[i])**2 for i in range(n)])


def rosenbrockchain(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError('dimension must be greater one')
    #return np.sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(self.n-1)])
    return np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)


def const_sphere(x):
    # print("x:{}".format(x))
    # sys.exit(0)
    if np.any(x < 0.0):
        return np.inf
    else:
        return sphere(x)


def const_ktablet(x):
    if np.any(x < 0.0):
        return np.inf
    else:
        return ktablet(x)


def const_ellipsoid(x):
    if np.any(x < 0.0):
        return np.inf
    else:
        return ellipsoid(x)


def const_rosen(x):
    if np.any(x > 1.0):
        return np.inf
    else:
        return rosenbrockchain(x)


def rastrigin(x):
    n = len(x)
    if n < 2:
        raise ValueError('dimension must be greater one')
    return 10*n + sum(x**2 - 10*np.cos(2*np.pi*x))


def ackley(x):
    n = len(x)
    f_value = 20.0
    tmp1 = np.sum([pow(x[i], 2.0) for i in range(n)])
    tmp2 = np.sum([np.cos(2. * np.pi * x[i]) for i in range(n)])
    f_value -= 20. * np.exp(-0.2 * np.sqrt(tmp1 / n))
    f_value += np.exp(1.0)
    f_value -= np.exp(tmp2 / n)
    return f_value


def bohachevsky(x):
    n = len(x)
    f_value = 0.
    for i in range(n - 1):
        f_value += pow(x[i], 2.0)
        f_value += 2 * pow(x[i + 1], 2.0)
        f_value -= 0.3 * np.cos(3 * np.pi * x[i])
        f_value -= 0.4 * np.cos(4 * np.pi * x[i + 1])
        f_value += 0.7

    return f_value


def schaffer(x):
    n = len(x)
    return np.sum([pow(pow(x[i], 2.0) + pow(x[i+1], 2.0), 0.25) * (pow(np.sin(50 * pow(pow(x[i], 2.0) + pow(x[i + 1], 2.0), 0.1)), 2.0) + 1.0)
                   for i in range(n - 1)])


