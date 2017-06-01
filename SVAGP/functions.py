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
from settings import float_type
from quadrature import hermgauss
import numpy as np


def eye(N):
    """
    An identitiy matrix
    """
    return tf.diag(tf.ones(tf.stack([N, ]), dtype=float_type))


def variational_expectations( Fmu, Fvar, phi, num_gauss_hermite_points=20):
    """
    Compute the expected value of a function phi, given a Gaussian
    distribution for the input values.
    if
        q(f) = N(Fmu, Fvar)
    then this method computes
       \int phi(f) q(f) df.
    Here, we implement a default Gauss-Hermite quadrature routine
    """
    gh_x, gh_w = hermgauss(num_gauss_hermite_points)
    gh_x = gh_x.reshape(1, -1)
    gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
    shape = tf.shape(Fmu)
    Fmu, Fvar = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar)]
    X = gh_x * tf.sqrt(2.0 * Fvar) + Fmu
    logp = phi(X)
    return tf.reshape(tf.matmul(logp, gh_w), shape)


