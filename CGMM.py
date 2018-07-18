import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import sqrtm
from enum import Enum
import time
from sklearn.cluster import KMeans


class Convex_GMM(object):
    """ Convex_GMM model
    """

    eps = 1e-1

    def __init__(self, dimension=1, n_components=1, batch_size=200, step=1e-2):

        # samples dimension
        self.d = dimension

        # number of modes
        self.k = n_components

        # normal covariance structure
        self.bs = batch_size

        # input batch
        self.batch = tf.placeholder(tf.float32, [self.bs, self.d])


        #tf objects
        self.r_covs = None
        self.means = None
        self.weights = None
        self.objective = None
        self.obj_step = None

        self.build_fit_nn(step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        #pgd best so far
        # self.best_obj = float('-inf')
        # self.best_r_covs.load(self.r_covs, self.sess)
        # self.best_means.load(self.means, self.sess)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.sess.close()

    def build_fit_nn(self, step):
        # optimization variables
        self.r_covs = tf.Variable(tf.truncated_normal([self.k, self.d, self.d], stddev=0.1))
        self.means = tf.Variable(tf.truncated_normal([self.k, self.d], stddev=0.1))
        self.weights = tf.Variable(np.random.dirichlet(np.ones(self.k), self.bs), dtype='float32')

        # self.objective = tf.constant(.0)

        Z = self.batch - tf.matmul(self.weights, self.means)
        Sigma = tf.einsum('ijk,ilk->ijl', self.r_covs, self.r_covs)
        Q = tf.einsum('li,ijk->ljk', tf.square(self.weights), Sigma)
        self.objective = tf.einsum('ji,lik,jk->', Z, tf.matrix_inverse(Q), Z) +\
                         tf.reduce_sum(tf.log(tf.matrix_determinant(Q)))

        # for i in range(self.bs):
        #     z = self.batch[i] - tf.matmul(tf.reshape(self.weights[i], (1, -1)), self.means)
        #     Q = tf.einsum('i,ijk->jk', tf.square(self.weights[i]), tf.einsum('ijk,ilk->ijl', self.r_covs, self.r_covs))
        #     self.objective += tf.matmul(tf.matmul(z, tf.matrix_inverse(Q)), z, transpose_b=True) + tf.log(tf.matrix_determinant(Q))

        self.obj_step = tf.train.AdamOptimizer(step).minimize(self.objective)

        # projection step
        self.proj_weights = self.Projection(self.Projection.ProjType.SIMPLEX, self.weights, self.bs, self.k)


    def store_params(self):
        self.means.load(means, self.sess)
        self.bes_r_covs.load(self.r_covs, self.sess)
        self.best_means.load(self.means, self.sess)

    def set_params(self, means=None, covs=None):
        """
        :param means:
        :param covs:
        :return:
        """
        if means is not None:
            self.means.load(means, self.sess)
        if covs is not None:
            self.r_covs.load(np.array([sqrtm(cov) for cov in covs]), self.sess)

    def sample(self, n_samples=100, means=None, covs=None, sparsity=0.2):
        """ Samples from convex combination of GM

        :param means: kxd array of means
        :param convs: kxdxd covariance matrices
        :param n_samples: number of samples
        :return: n_samples of CGMM according to the parameters
        """

        if means is None:
            means = self.sess.run(self.means)
        if covs is None:
            covs = self.sess.run(tf.einsum('ijk,ilk->ijl', self.r_covs, self.r_covs))

        samples = np.dstack([np.random.multivariate_normal(mean, cov, n_samples)
                             for (mean, cov) in zip(means, covs)])

        weights = np.random.dirichlet(np.ones(self.k)*sparsity, n_samples)
        weights = np.tile(weights.reshape(n_samples, 1, self.k),(1, self.d, 1))
        return np.sum(np.multiply(samples, weights), 2)

    def train_step_pgd(self, X):
        """ fits data set to CGMMM model

        :param X: NxD array of points
        :returns: (means, covs) explains the samples as CGMM model
        """
        self.sess.run(self.obj_step, feed_dict={self.batch: X})
        self.proj_weights.project(self.sess)

    def train_pgd(self, X, epoch=10000):
        for i in range(epoch):
            self.train_step_pgd(X)
            if i%1000 == 0:
                print(self.get_params())

            # if self.sess.run(self.objective, feed_dict={self.batch: X}) < self.best_obj:
            #     self.store_params()

    def get_params(self):
        return self.sess.run([self.means, tf.einsum('ijk,ilk->ijl', self.r_covs, self.r_covs)])

    def visualize(self, X=None):
        """ In  case X is 2 dim data, visualizes the samples (X) with the model density contours
        """
        n_samples = 1000

        plt.scatter(X[:, 0], X[:, 1])

        means, covs = self.get_params()

        x_max, y_max = np.max(means,0) + (np.diag(np.max(covs,0)) * 2)
        x_min, y_min = np.min(means,0) - (np.diag(np.max(covs,0)) * 2)
        x_jmp = (x_max - x_min) / n_samples
        y_jmp = (y_max - y_min) / n_samples

        x, y = np.mgrid[x_min:x_max:x_jmp, y_min:y_max:y_jmp]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        rvs = [mvn(means[i], covs[i]) for i in range(self.k)]
        plt.contour(x, y, logsumexp([rvs[i].logpdf(pos) + 0.5 for i in range(len(rvs))],0))

        plt.show()

    class Projection(object):
        """ Convex_GMM model
        """
        class ProjType(Enum):
            SIMPLEX = 1
            PSD = 2

        def __init__(self, proj_type, X, n, d):
            self.proj_type = proj_type
            self.X = X
            if self.proj_type is self.ProjType.SIMPLEX:
                self.proj_x, self.x_init, self.opt_step, self.opt_stop = self.simplex_proj_nn(self.X, n, d)

        def simplex_proj_nn(self, X, n, d):
            """
            return tensor operation represents projection on simplex
            :param X:
            :return:
            """
            dual = tf.Variable(tf.truncated_normal([n, 1], stddev=0.1))
            ones = tf.constant([[1.]]*d)

            dual_obj = -0.5*tf.reduce_sum(tf.square(tf.nn.relu(tf.matmul(dual, ones, transpose_b=True) - X)), axis=1) - \
                       dual * (tf.reduce_sum(X,axis=1, keep_dims=True) - 1) + \
                       0.5*d*tf.square(dual)
            proj_val = tf.assign(X, tf.nn.relu(X - tf.matmul(dual, ones, transpose_b=True)))

            u = tf.Variable(tf.ones([n,1]))
            l = tf.Variable(tf.ones([n,1]))
            r_max = tf.reduce_max(X,axis=1, keep_dims=True)
            set_bounds = tf.group(
                tf.assign(l,  r_max - 1),
                tf.assign(u, r_max),
                tf.assign(dual, r_max - 0.5))

            pos = tf.less(tf.gradients(dual_obj, dual)[0], 0)
            bisection_step =  tf.group(tf.assign(l, tf.where(pos, (u + l)/2, l)),
                                       tf.assign(u, tf.where(pos, u, (u + l)/2)),
                                       tf.assign(dual, tf.where(pos, (dual + u)/2, (dual + l)/2)))

            bisection_stop = tf.reduce_all(tf.less(1e-4, u-l))

            return proj_val, set_bounds, bisection_step, bisection_stop

        def project(self, sess):
            if self.proj_type is self.ProjType.SIMPLEX:
                sess.run(self.x_init)
                while sess.run(self.opt_stop):
                    sess.run(self.opt_step)
                sess.run(self.proj_x)


if __name__ == "__main__":
    means = np.vstack([[0, 10], np.ones(2) * 10, np.zeros(2)])
    covs = np.vstack([np.eye(2), np.eye(2), np.eye(2)]).reshape(3, 2, 2)

    x = None
    bs = 1000
    dim = 2
    comp = 3
    step = 1e-2
    epoch = 10000
    with Convex_GMM(dim, comp, bs, step) as c:

        # for i in range(500):
        x = c.sample(bs, means, covs)
        start = time.time()
        c.train_pgd(x, epoch)
        print('elapsed: ', (time.time() - start)/60)

        print(c.get_params())
        c.visualize(x)

