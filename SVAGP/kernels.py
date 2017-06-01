# Copyright 2016 James Hensman, alexggmatthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------
# Modification notice:
# This file was modified by Vincent ADAM
# ------------------------------------------

import tensorflow as tf
import numpy as np
from settings import int_type



class Kern(object):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None):
        """
        input dim is an integer
        active dims is either an iterable of integers or None.

        Input dim is the number of input dimensions to the kernel. If the
        kernel is computed on a matrix X which has more columns than input_dim,
        then by default, only the first input_dim columns are used. If
        different columns are required, then they may be specified by
        active_dims.

        If active dims is None, it effectively defaults to range(input_dim),
        but we store it as a slice for efficiency.
        """
        self.input_dim = int(input_dim)
        if active_dims is None:
            self.active_dims = slice(input_dim)
        elif type(active_dims) is slice:
            self.active_dims = active_dims
            if active_dims.start is not None and active_dims.stop is not None and active_dims.step is not None:
                assert len(range(*active_dims)) == input_dim  # pragma: no cover
        else:
            self.active_dims = np.array(active_dims, dtype=np.int32)
            assert len(active_dims) == input_dim

        self.num_gauss_hermite_points = 20

    def _slice(self, X, X2):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims`.
        :param X: Input 1 (NxD).
        :param X2: Input 2 (MxD), may be None.
        :return: Sliced X, X2, (Nxself.input_dim).
        """
        if isinstance(self.active_dims, slice):
            X = X[:, self.active_dims]
            if X2 is not None:
                X2 = X2[:, self.active_dims]
        else:
            X = tf.transpose(tf.gather(tf.transpose(X), self.active_dims))
            if X2 is not None:
                X2 = tf.transpose(tf.gather(tf.transpose(X2), self.active_dims))
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(X)[1], tf.constant(self.input_dim, dtype=int_type))
        ]):
            X = tf.identity(X)

        return X, X2

    def _slice_cov(self, cov):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims` for covariance matrices. This requires slicing the
        rows *and* columns. This will also turn flattened diagonal
        matrices into a tensor of full diagonal matrices.
        :param cov: Tensor of covariance matrices (NxDxD or NxD).
        :return: N x self.input_dim x self.input_dim.
        """
        cov = tf.cond(tf.equal(tf.rank(cov), 2), lambda: tf.matrix_diag(cov), lambda: cov)

        if isinstance(self.active_dims, slice):
            cov = cov[..., self.active_dims, self.active_dims]
        else:
            cov_shape = tf.shape(cov)
            covr = tf.reshape(cov, [-1, cov_shape[-1], cov_shape[-1]])
            gather1 = tf.gather(tf.transpose(covr, [2, 1, 0]), self.active_dims)
            gather2 = tf.gather(tf.transpose(gather1, [1, 0, 2]), self.active_dims)
            cov = tf.reshape(tf.transpose(gather2, [2, 0, 1]),
                             tf.concat_v2([cov_shape[:-2], [len(self.active_dims), len(self.active_dims)]], 0))
        return cov


class Stationary(Kern):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=1.,
                 active_dims=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        """
        Kern.__init__(self, input_dim, active_dims)
        self.lengthscales = tf.get_variable("lengthscales", [input_dim],  initializer=tf.constant_initializer(lengthscales))
        self.variance = tf.get_variable("variance", [1],  initializer=tf.constant_initializer(variance))


    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, tf.transpose(X)) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, tf.transpose(X2)) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)



class PeriodicKernel(Kern):
    """
    The periodic kernel. Defined in  Equation (47) of
    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.
    Derived using the mapping u=(cos(x), sin(x)) on the inputs.
    """

    def __init__(self, input_dim, period=1.0, variance=1.0,
                 lengthscales=1.0, active_dims=None):
        Kern.__init__(self, input_dim, active_dims)
        self.lengthscales = tf.get_variable("lengthscales", [input_dim],  initializer=tf.constant_initializer(lengthscales))
        self.variance = tf.get_variable("variance", [1],  initializer=tf.constant_initializer(variance))
        self.period = tf.get_variable("period", [1],  initializer=tf.constant_initializer(period))

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D

        r = np.pi * (f - f2) / self.period
        r = tf.reduce_sum(tf.square(tf.sin(r) / self.lengthscales), 2)

        return self.variance * tf.exp(-0.5 * r)


class LocallyPeriodicKernel(Kern):
    """
    k(t) = var *  exp ( - t^2 / len^2 ) * cos ( 2 * pi * t / per )
    """

    def __init__(self, input_dim, period=1.0, variance=1.0,
                 lengthscales=1.0, active_dims=None):
        Kern.__init__(self, input_dim, active_dims)
        self.lengthscales = tf.get_variable("lengthscales", [input_dim],  initializer=tf.constant_initializer(lengthscales))
        self.variance = tf.get_variable("variance", [1],  initializer=tf.constant_initializer(variance))
        self.period = tf.get_variable("period", [1],  initializer=tf.constant_initializer(period))


    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X
        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        r = tf.reduce_sum(f-f2,2) #hack for 1d
        return self.variance * tf.exp( - tf.square(r/self.lengthscales) ) * tf.cos(2.*np.pi *r/ self.period)
