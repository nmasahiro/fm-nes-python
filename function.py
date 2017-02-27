import numpy as np
import math

class SphereFunction(object):
    """
    The Sphere function

    Attribute
    ---------
    n : int
        次元数
    name : str
        関数名

    Methods
    -------
    evaluate(x)
        xの評価を行う．
    get_optimal_solution()
        最適解の取得．
    """
    def __init__(self, n):
        self.n = n
        self.name = "%dD-SphereFunction" % n

    def evaluate(self, x):
        """
        Return a evaluation value

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            evaluation value
        """
        return np.sum(x**2)

    def get_optimal_solution(self):
        """
        Return a optimal solution.

        Returns
        -------
        ndarray
            optimal solution.
        """
        return np.zeros([self.n, 1])


class KTabletFunction(object):
    """
    The k-tablet function

    Attribute
    ---------
    n : int
        次元数
    k : int
        タブレット数
    name : str
        関数名

    Methods
    -------
    evaluate(x)
        xの評価を行う．
    get_optimal_solution()
        最適解の取得．
    """
    def __init__(self, n, k=None):
        self.n = n
        if k is None:
            self.k = int(n/4) # default k value
        else:
            self.k = k
        self.name = "%dD-%d-tabletFunction" % (n, self.k)

    def evaluate(self, x):
        """
        Return a evaluate value

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            evaluation value
        """
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        #return np.sum(x[0:self.k]**2) + np.sum((100*x[self.k:self.n])**2)
        return x[0]**2 + (100*x[1])**2

    def get_optimal_solution(self):
        """
        Return a optimal solution.

        Returns
        -------
        ndarray
            optimal solution.
        """
        return np.zeros([self.n, 1])


class EllipsoidFunction(object):
    """
    The Ellipsoid function

    Attribute
    ---------
    n : int
        次元数
    name : str
        関数名

    Methods
    -------
    evaluate(x)
        xの評価を行う．
    get_optimal_solution()
        最適解の取得．
    """
    def __init__(self, n):
        self.n = n
        self.name = "%dD-EllipsoidFunction" % n
        self.aratio = 1000**(np.arange(n)/(n-1)).reshape(n, 1)

    def evaluate(self, x):
        """
        Return a evaluate value

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            evaluation value
        """
        val = 0.
        for i in range(self.n):
            val += 1000.**(i/(self.n-1.)) * x[i] * 1000.**(i/(self.n-1.)) * x[i]
        return val
        # return np.sum([(1000**(i / (self.n-1)) * x[i])**2 for i in range(self.n)])
        # return np.sum((self.aratio * x)**2)

    def get_optimal_solution(self):
        """
        Return a optimal solution.

        Returns
        -------
        ndarray
            optimal solution.
        """
        return np.zeros([self.n, 1])

class ConstraintSphereFunction(object):
    def __init__(self, n):
        self.n = n
        self.name = "%dD-ConstraintSphereFunction" % n
    def evaluate(self, x):
        if len(list(filter(lambda a: a < 0., x))) >= 1:
            return math.inf
        else:
            return np.sum(x**2)


class RosenbrockChainFunction(object):
    """
    The Rosenbrock(chain) function

    Attribute
    ---------
    n : int
        次元数
    name : str
        関数名

    Methods
    -------
    evaluate(x)
        xの評価を行う．
    get_optimal_solution()
        最適解の取得．
    """
    def __init__(self, n):
        self.n = n
        self.name = "%dD-RosenbrockChainFunction" % n

    def evaluate(self, x):
        """
        Return a evaluate value

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            evaluation value
        """
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        #return np.sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(self.n-1)])
        return np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

    def get_optimal_solution(self):
        """
        Return a optimal solution.

        Returns
        -------
        ndarray
            optimal solution.
        """
        return np.ones([self.n, 1])


class RosenbrockStarFunction(object):
    """
    The Rosenbrock(star) function

    Attribute
    ---------
    n : int
        次元数
    name : str
        関数名

    Methods
    -------
    evaluate(x)
        xの評価を行う．
    get_optimal_solution()
        最適解の取得．
    """
    def __init__(self, n):
        self.n = n
        self.name = "%dD-RosenbrockStarFunction" % n

    def evaluate(self, x):
        """
        Return a evaluate value

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            evaluation value
        """
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        #return np.sum([100*(x[0] - x[i+1]**2)**2 + (x[i+1] - 1)**2 for i in range(self.n-1)])
        return np.sum(100.0*(x[1:]**2 - x[0])**2 + (x[1:] - 1)**2)

    def get_optimal_solution(self):
        """
        Return a optimal solution.

        Returns
        -------
        ndarray
            optimal solution.
        """
        return np.ones([self.n, 1])


class AckleyFunction(object):
    """
    The Ackley function

    Attribute
    ---------
    n : int
        次元数
    name : str
        関数名

    Methods
    -------
    evaluate(x)
        xの評価を行う．
    get_optimal_solution()
        最適解の取得．
    """
    def __init__(self, n):
        self.n = n
        self.name = "%dD-AckleyFunction" % d

    def evaluate(self, x):
        """
        Return a evaluate value

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            evaluation value
        """
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        return np.sum(x[:-1]**2 + 2*x[1:]**2 - 0.3*np.cos(3*np.pi*x[:-1]) \
                - 0.4*np.cos(4*np.pi*x[1:]) + 0.7)

    def get_optimal_solution(self):
        """
        Return a optimal solution.

        Returns
        -------
        ndarray
            optimal solution.
        """
        return np.zeros([self.n, 1])


class SchafferFunction(object):
    """
    The Schaffer function

    Attribute
    ---------
    n : int
        次元数
    name : str
        関数名

    Methods
    -------
    evaluate(x)
        xの評価を行う．
    get_optimal_solution()
        最適解の取得．
    """
    def __init__(self, n):
        self.n = n
        self.name = "%dD-SchafferFunction" % n

    def evaluate(self, x):
        """
        Return a evaluate value

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            evaluation value
        """
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        y = x[:-1]**2 + x[1:]**2
        return np.sum(y**0.25 * (np.sin(50*y**0.1)**2 + 1.0))

    def get_optimal_solution(self):
        """
        Return a optimal solution.

        Returns
        -------
        ndarray
            optimal solution.
        """
        return np.zeros([self.n, 1])


class RastriginFunction(object):
    """
    The Rastrigin function

    Attribute
    ---------
    n : int
        次元数
    name : str
        関数名

    Methods
    -------
    evaluate(x)
        xの評価を行う．
    get_optimal_solution()
        最適解の取得．
    """
    def __init__(self, n):
        self.n = n
        self.name = "%dD-RastriginFunction" % n

    def evaluate(self, x):
        """
        Return a evaluate value

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            evaluation value
        """
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        return 10*self.n + sum(x**2 - 10*np.cos(2*np.pi*x))

    def get_optimal_solution(self):
        """
        Return a optimal solution.

        Returns
        -------
        ndarray
            optimal solution.
        """
        return np.zeros([self.n, 1])

