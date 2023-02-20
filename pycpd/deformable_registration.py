from builtins import super
import time
import numpy as np
from numba import njit
import numbers
from .emregistration import EMRegistration
from .utility import gaussian_kernel, low_rank_eigen

class DeformableRegistration(EMRegistration):
    """
    Deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.
    
    low_rank: bool
        Whether to use low rank approximation.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(self, alpha=None, beta=None, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))

        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            print("Calculating low rank")
            st = time.time()
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
            self.E = 0.
            et = time.time()
            print(f"Calculating low rank took {et - st} seconds.")

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            # A = np.dot(np.diag(self.P1), self.G) + \
            #     self.alpha * self.sigma2 * np.eye(self.M)
            # B = self.PX - np.dot(np.diag(self.P1), self.Y)
            # self.W = np.linalg.solve(A, B)
            self.W = self._update_transform(self.P1, self.G, self.M, self.sigma2, self.alpha, self.Y, self.PX)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            # dP = np.diag(self.P1)
            # dPQ = np.matmul(dP, self.Q)
            # F = self.PX - np.matmul(dP, self.Y)

            # self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
            #     np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
            #                     (np.matmul(self.Q.T, F))))))
            # QtW = np.matmul(self.Q.T, self.W)
            # self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))
            self.W, self.E = self._update_transform_low(self.P1, self.sigma2, self.alpha, self.Y, self.PX, self.Q, self.E, self.inv_S, self.S)

    @staticmethod
    @njit
    def _update_transform(P1, G, M, sigma2, alpha, Y, PX):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        A = np.dot(np.diag(P1), G) + \
            alpha * sigma2 * np.eye(M)
        B = PX - np.dot(np.diag(P1), Y)
        W = np.linalg.solve(A, B)
        return W


    @staticmethod
    @njit
    def _update_transform_low(P1, sigma2, alpha, Y, PX, Q, E, inv_S, S):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        # Matlab code equivalent can be found here:
        # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
        dP = np.diag(P1)
        dPQ = dP @ Q
        F = PX - dP @ Y
        
        W = 1 / (alpha * sigma2) * (F - (dPQ @ (np.linalg.solve((alpha * sigma2 * inv_S + (Q.T @ dPQ)),((Q.T @ F))))))
        QtW = Q.T @ W
        E = E + alpha / 2 * np.trace((QtW.T @ (S @ QtW)))
        return W, E

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                

        """
        if Y is not None:
            # G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            # return Y + np.dot(G, self.W)
            return self._transform_point_cloud(Y, self.beta, self.Y, self.W)
        else:
            if self.low_rank is False:
                # self.TY = self.Y + np.dot(self.G, self.W)
                self.TY = self._transform_y(self.Y, self.G, self.W)

            elif self.low_rank is True:
                # self.TY = self.Y + np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))
                self.TY = self._transform_y_low(self.Y, self.Q, self.S, self.W)
                return

    @staticmethod
    @njit
    def _transform_point_cloud(X, beta, Y, W):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                

        """
        G = gaussian_kernel(X=X, beta=beta, Y=Y)
        return Y + np.dot(G, W)

    @staticmethod
    @njit
    def _transform_y(Y, G, W):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                

        """
        return Y + np.dot(G, W)

    @staticmethod
    @njit
    def _transform_y_low(Y, Q, S, W):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                

        """
        return Y + (Q @ (S @ (Q.T @ W)))

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = np.inf

        # xPx = np.dot(np.transpose(self.Pt1), np.sum(
        #     np.multiply(self.X, self.X), axis=1))
        # yPy = np.dot(np.transpose(self.P1),  np.sum(
        #     np.multiply(self.TY, self.TY), axis=1))
        # trPXY = np.sum(np.multiply(self.TY, self.PX))

        # self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)
        self.sigma2 = self._update_variance(self.Pt1, self.X, self.P1, self.TY, self.PX, self.Np, self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = np.abs(self.sigma2 - qprev)

    @staticmethod
    @njit
    def _update_variance(Pt1, X, P1, TY, PX, Np, D):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """

        xPx = np.dot(np.transpose(Pt1), np.sum(
            np.multiply(X, X), axis=1))
        yPy = np.dot(np.transpose(P1),  np.sum(
            np.multiply(TY, TY), axis=1))
        trPXY = np.sum(np.multiply(TY, PX))

        return (xPx - 2 * trPXY + yPy) / (Np * D)


    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.


        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W
