import tensorflow as tf
import numpy as np
from settings import float_type, jitter_level,std_qmu_init, np_float_type, np_int_type
from functions import eye, variational_expectations
from mean_functions import Zero
from kullback_leiblers import gauss_kl_white, gauss_kl_white_diag, gauss_kl, gauss_kl_diag
from conditionals import conditional
from quadrature import hermgauss


class ChainedGPs(object):
    """
    Chained Gaussian Processes

    The key reference for this algorithm is:
    ::
      @article{saul2016chained,
        title={Chained Gaussian Processes},
        author={Saul, Alan D and Hensman, James and Vehtari, Aki and Lawrence, Neil D},
        journal={arXiv preprint arXiv:1604.05263},
        year={2016}
      }
    
    """
    def __init__(self, X, Y, kerns,likelihood,Zs,mean_functions=None, whiten=True,q_diag=False, f_indices=None):
        '''
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kerns, likelihood, mean_functions are appropriate (single or list of) GPflow objects
        - Zs is a list of  matrices of pseudo inputs, size M[k] x C
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        '''
        self.likelihood = likelihood
        self.kerns = kerns
        self.C = len(kerns)
        self.mean_functions = [Zero() for _ in range(self.C)] if mean_functions is None else mean_functions
        self.f_indices = f_indices # function of one variable
        self.X = X
        self.Y = Y
        self.Zs = Zs
        self.num_inducing = [z.get_shape()[0] for z in Zs]
        self.num_latent = Y.get_shape()[-1]
        self.num_data = Y.get_shape()[0]
        self.whiten=whiten
        self.q_diag = q_diag
        self.initialize_inference()

    def initialize_inference(self):

        with tf.variable_scope("inference") as scope:
            self.q_mu,self.q_sqrt = [],[]
            for c in range(self.C):

                self.q_mu.append( tf.get_variable("q_mu%d"%c,[self.num_inducing[c], self.num_latent],\
                          initializer=tf.constant_initializer(np.random.randn(self.num_inducing[c], self.num_latent)*std_qmu_init,\
                                                              dtype=float_type)))

                if self.q_diag:
                    q_sqrt = np.ones((self.num_inducing[c], self.num_latent))
                    self.q_sqrt.append( tf.get_variable("q_sqrt%d"%c,[self.num_inducing[c],self.num_latent], \
                                          initializer=tf.constant_initializer(q_sqrt,dtype=float_type)) )

                else:
                    q_sqrt = np.array([np.eye(self.num_inducing[c]) for _ in range(self.num_latent)]).swapaxes(0, 2)
                    self.q_sqrt.append( tf.get_variable("q_sqrt%d"%c,[self.num_inducing[c],self.num_inducing[c],self.num_latent], \
                                          initializer=tf.constant_initializer(q_sqrt,dtype=float_type)) )

    def build_prior_KL(self):

        KL = tf.Variable(0,name='KL',trainable=False,dtype=float_type)
        for i in range(self.C):
            if self.whiten:
                if self.q_diag:
                    KL += gauss_kl_white_diag(self.q_mu[i], self.q_sqrt[i])
                else:
                    KL += gauss_kl_white(self.q_mu[i], self.q_sqrt[i])
            else:
                K = self.kerns[i].K(self.Zs[self.f_indices[i]]) + eye(self.num_inducing[i]) * jitter_level
                if self.q_diag:
                    KL += gauss_kl_diag(self.q_mu[i], self.q_sqrt[i], K)
                else:
                    KL += gauss_kl(self.q_mu[i], self.q_sqrt[i], K)
        return KL

    def get_covariate(self,Xnew,c):
        return tf.transpose(tf.gather(tf.transpose(Xnew),self.f_indices[c]))

    def build_predict_fs(self, Xnew):
        mus, vars = [],[]
        for c in range(self.C):
            x = self.get_covariate(Xnew,c)
            mu, var = conditional(x, self.Zs[c], self.kerns[c], self.q_mu[c],
                                     q_sqrt=self.q_sqrt[c], full_cov=False, whiten=self.whiten)
            mus.append(mu+self.mean_functions[c](x))
            vars.append(var)
        return tf.stack(mus),tf.stack(vars)


class SVAGP(ChainedGPs):
    """
    Sparse Variational Additive Gaussian Process
    
    Key reference is 
    ::
      @inproceedings{adam2016scalable,
        title={Scalable transformed additive signal decomposition by non-conjugate Gaussian process inference},
        author={Adam, Vincent and Hensman, James and Sahani, Maneesh},
        booktitle={Machine Learning for Signal Processing (MLSP), 2016 IEEE 26th International Workshop on},
        pages={1--6},
        year={2016},
        organization={IEEE}
      }    
    """

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """
        cost = -self.build_prior_KL()
        fmean, fvar = self.build_predict_additive_predictor(self.X)
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        cost += tf.reduce_sum(var_exp)
        return cost


    def build_predict_additive_predictor(self, Xnew):
        mus, vars = self.build_predict_fs(Xnew)
        return tf.reduce_sum(mus,0),tf.reduce_sum(vars,0)

