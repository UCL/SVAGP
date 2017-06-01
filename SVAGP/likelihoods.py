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

from densities import gaussian, poisson, bernoulli
import tensorflow as tf
import numpy as np
from quadrature import hermgauss


class Likelihood(object):
    def __init__(self):
        self.num_gauss_hermite_points = 20

    def logp(self, F, Y):
        """
        Return the log density of the data given the function values.
        """
        raise NotImplementedError("implement the logp function\
                                  for this likelihood")

    def conditional_mean(self, F):
        """
        Given a value of the latent function, compute the mean of the data

        If this object represents

            p(y|f)

        then this method computes

            \int y p(y|f) dy
        """
        raise NotImplementedError

    def conditional_variance(self, F):
        """
        Given a value of the latent function, compute the variance of the data

        If this object represents

            p(y|f)

        then this method computes

            \int y^2 p(y|f) dy  - [\int y p(y|f) dy] ^ 2

        """
        raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_w /= np.sqrt(np.pi)
        gh_w = gh_w.reshape(-1, 1)
        shape = tf.shape(Fmu)
        Fmu, Fvar = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar)]
        X = gh_x[None, :] * tf.sqrt(2.0 * Fvar) + Fmu

        # here's the quadrature for the mean
        E_y = tf.reshape(tf.matmul(self.conditional_mean(X), gh_w), shape)

        # here's the quadrature for the variance
        integrand = self.conditional_variance(X) \
            + tf.square(self.conditional_mean(X))
        V_y = tf.reshape(tf.matmul(integrand, gh_w), shape) - tf.square(E_y)

        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

           \int p(y=Y|f)q(f) df

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)

        gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
        shape = tf.shape(Fmu)
        Fmu, Fvar, Y = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y)]
        X = gh_x[None, :] * tf.sqrt(2.0 * Fvar) + Fmu

        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])  # broadcast Y to match X

        logp = self.logp(X, Y)
        return tf.reshape(tf.log(tf.matmul(tf.exp(logp), gh_w)), shape)

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """

        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x.reshape(1, -1)
        gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
        shape = tf.shape(Fmu)
        Fmu, Fvar, Y = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y)]
        X = gh_x * tf.sqrt(2.0 * Fvar) + Fmu
        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])  # broadcast Y to match X

        logp = self.logp(X, Y)
        return tf.reshape(tf.matmul(logp, gh_w), shape)


class Gaussian(Likelihood):
    def __init__(self,variance=1):
        Likelihood.__init__(self)
        self.variance = tf.get_variable("variance", [1],  initializer=tf.constant_initializer(variance))

    def logp(self, F, Y):
        return gaussian(F, Y, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return gaussian(Fmu, Y, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance


class Gaussian_with_link(Likelihood):

    def __init__(self,variance=1,invlink=None):
        Likelihood.__init__(self)
        self.variance = tf.get_variable("variance", [1],  initializer=tf.constant_initializer(variance))
        self.invlink = invlink

    def logp(self, F, Y):
        return gaussian(self.invlink(F), Y, self.variance)

    def conditional_mean(self, F):
        return tf.identity(self.invlink(F))

    def conditional_variance(self, F):
        return tf.fill(tf.shape(self.invlink(F)), tf.squeeze(self.variance))






class Poisson(Likelihood):
    def __init__(self, invlink=tf.exp):
        Likelihood.__init__(self)
        self.invlink = invlink

    def logp(self, F, Y):
        return poisson(self.invlink(F), Y)

    def conditional_variance(self, F):
        return self.invlink(F)

    def conditional_mean(self, F):
        return self.invlink(F)

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return Y * Fmu - tf.exp(Fmu + Fvar / 2) - tf.lgamma(Y + 1)
        else:
            return Likelihood.variational_expectations(self, Fmu, Fvar, Y)


def probit(x):
    return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2e-3) + 1e-3


class Bernoulli(Likelihood):
    def __init__(self, invlink=probit):
        Likelihood.__init__(self)
        self.invlink = invlink

    def logp(self, F, Y):
        return bernoulli(self.invlink(F), Y)

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is probit:
            p = probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return Likelihood.predict_mean_and_var(self, Fmu, Fvar)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return bernoulli(p, Y)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.invlink(F)
        return p - tf.square(p)

