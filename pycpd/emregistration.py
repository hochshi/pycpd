from __future__ import division
import numpy as np
from numba import njit
import time
import numbers
from warnings import warn
import pyfgt

@njit
def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).

    Attributes
    ----------
    X: numpy array
        NxD array of points for target.
    
    Y: numpy array
        MxD array of points for source.
    
    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = np.expand_dims(X, axis=0) - np.expand_dims(Y, 1)
    err = diff ** 2
    return np.sum(err) / (D * M * N)

def lowrankQS(G, beta, num_eig, eig_fgt=False):
    """
    Calculate eigenvectors and eigenvalues of gaussian matrix G.
    
    !!!
    This function is a placeholder for implementing the fast
    gauss transform. It is not yet implemented.
    !!!

    Attributes
    ----------
    G: numpy array
        Gaussian kernel matrix.
    
    beta: float
        Width of the Gaussian kernel.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation of G
    
    eig_fgt: bool
        If True, use fast gauss transform method to speed up. 
    """

    # if we do not use FGT we construct affinity matrix G and find the
    # first eigenvectors/values directly

    if eig_fgt is False:
        S, Q = np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.

        return Q, S

    elif eig_fgt is True:
        raise Exception('Fast Gauss Transform Not Implemented!')

class EMRegistration(object):
    """
    Expectation maximization point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    TY: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    """

    def __init__(self, X, Y, sigma2=None, max_iterations=None, tolerance=None, w=None, fgt=False, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

        self.X = X
        self.Y = Y
        self.TY = Y
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.Pt1 = np.zeros((self.N, ))
        self.P1 = np.zeros((self.M, ))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0
        self.fgt = fgt

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.

        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.
        
        Returns
        -------
        self.TY: numpy array
            MxD array of transformed source points.
        
        registration_parameters:
            Returned params dependent on registration method used. 
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        st = time.time()
        self.expectation()
        et = time.time()
        print(f"Expectation took {et - st} seconds.")
        st = time.time()
        self.maximization()
        et = time.time()
        print(f"maximization took {et - st} seconds.")
        st = time.time()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        if not self.fgt:
            P = self._full_P(self.X, self.TY, self.sigma2)
        else:
            # print("Calculating P fgt")
            # st = time.time()
            P = self._fgt_P(self.X, self.TY, self.sigma2)
            # et = time.time()
            # print(f"P fgt took {et - st} seconds.")
            # self.Pt1, self.P1, self.Np, self.PX = self._fgt_expectation(self.X, self.TY, self.sigma2, self.D, self.w, self.M, self.N)
        self.Pt1, self.P1, self.Np, self.PX = self._expectation(self.X, P, self.sigma2, self.D, self.w, self.M, self.N)

    @staticmethod
    def _expectation(X, P, sigma2, D, w, M, N):
        """
        Compute the expectation step of the EM algorithm.
        """
        # P = np.sum((np.expand_dims(X, axis=0) - np.expand_dims(TY, axis=1))**2, axis=2) # (M, N)
        # P = np.exp(-P/(2*sigma2))
        c = (2*np.pi*sigma2)**(D/2)*w/(1. - w)*M/N

        den = np.sum(P, axis = 0).reshape(1,-1) # (1, N)
        den = np.clip(den, np.finfo(X.dtype).eps, None) + c

        normed_P = np.divide(P, den)
        Pt1 = np.sum(normed_P, axis=0)
        P1 = np.sum(normed_P, axis=1)
        Np = np.sum(P1)
        PX = normed_P @ X
        return (Pt1, P1, Np, PX)

    @staticmethod
    @njit
    def _full_P(X, TY, sigma2):
        """
        Compute the probabily matrix for expectation step of the EM algorithm.
        """
        P = np.sum((np.expand_dims(X, axis=0) - np.expand_dims(TY, axis=1))**2, axis=2) # (M, N)
        P = np.exp(-P/(2*sigma2))
        return P

    @staticmethod
    def _fgt_P(X, TY, sigma2):
        bandwidth = np.sqrt(2*sigma2)
        return pyfgt.mat_direct_tree(X, TY, bandwidth)

    @staticmethod
    def _fgt_expectation(X, TY, sigma2, D, w, M, N):
        bandwidth = np.sqrt(2*sigma2)
        print("Calculating Kt1 fgt")
        st = time.time()
        Kt1 = pyfgt.direct_tree(TY, X, bandwidth)
        et = time.time()
        print(f"Kt1 fgt took {et - st} seconds.")
        c = (2*np.pi*sigma2)**(D/2)*w/(1. - w)*M/N
        a = np.divide(1.0, Kt1 + c)
        Pt1 = 1 - c * a
        P1 = pyfgt.wdirect3(TY, X, bandwidth, a)
        Np = np.sum(P1)
        PX0 = pyfgt.wdirect3(TY, X, bandwidth, a * X[:, 0])
        PX1 = pyfgt.wdirect3(TY, X, bandwidth, a * X[:, 1])
        PX2 = pyfgt.wdirect3(TY, X, bandwidth, a * X[:, 2])
        PX = np.hstack([PX0, PX1, PX2])
        return (Pt1, P1, Np, PX)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        st = time.time()
        self.update_transform()
        et = time.time()
        print(f"update_transform took {et - st} seconds.")
        st = time.time()
        self.transform_point_cloud()
        et = time.time()
        print(f"transform_point_cloud took {et - st} seconds.")
        st = time.time()
        self.update_variance()
        et = time.time()
        print(f"update_variance took {et - st} seconds.")
