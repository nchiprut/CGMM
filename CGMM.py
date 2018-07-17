import numpy as np
np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import sqrtm


class Convex_GMM(object):
    """ Convex_GMM model
    """

    eps = 1e-1

    def __init__(self, dimension=1, n_components=1, batch_size=200, sparsity=0.2):

        # samples dimension
        self.d = dimension

        # number of modes
        self.k = n_components

        # normal covariance structure
        self.bs = batch_size

        # sparsity factor between (0,1)
        self.sparsity = sparsity

        # input batch
        self.batch = tf.placeholder(tf.float32, [self.bs, self.d])

        #tf objects
        self.r_covs = None
        self.means = None
        self.weights = None
        self.objective = None
        self.obj_step = None

        self.build_fit_nn()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        """
        self.means = np.zeros([self.k, self.d])
        self.covs = np.zeros([self.k, self.d,self.d])
        self.covs[:,np.arange(self.d),np.arange(self.d)] = 1
        self.weights = None
        """

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.sess.close()

    def build_fit_nn(self):
        # optimization variables
        self.r_covs = tf.Variable(tf.truncated_normal([self.k, self.d, self.d], stddev=0.1))
        self.means = tf.Variable(tf.truncated_normal([self.k, self.d], stddev=0.1))
        self.weights = tf.Variable(np.random.dirichlet(np.ones(self.k), self.bs), dtype='float32')

        self.objective = tf.constant(.0)
        for i in range(self.bs):
            z = self.batch[i] - tf.matmul(tf.reshape(self.weights[i], (1, -1)), self.means)
            Q = tf.einsum('i,ijk->jk', tf.square(self.weights[i]), tf.einsum('ijk,ilk->ijl', self.r_covs, self.r_covs))
            self.objective += tf.matmul(tf.matmul(z, tf.matrix_inverse(Q)), z, transpose_b=True) + tf.log(tf.matrix_determinant(Q))

        self.obj_step = tf.train.AdamOptimizer().minimize(self.objective)

        # projection step
        self.proj_weights, self.weights_init, self.weights_step, self.weights_stop = self.simplex_poj(self.weights)


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

    def sample(self, n_samples=100, means=None, covs=None):
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

        weights = np.random.dirichlet(np.ones(self.k)*self.sparsity, n_samples)
        weights = np.tile(weights.reshape(n_samples, 1, self.k),(1, self.d, 1))
        return np.sum(np.multiply(samples, weights), 2)

    def train_step_gdp(self, X):
        """ fits data set to CGMMM model

        :param X: NxD array of points
        :returns: (means, covs) explains the samples as CGMM model
        """

        self.sess.run(self.obj_step, feed_dict={self.batch: X})
        self.sess.run(self.weights_init)
        while self.sess.run(self.weights_stop):
            self.sess.run(self.weights_step)
        self.sess.run(self.proj_weights)

    def train_gdp(self, X, epoch=30000):
        for i in range(epoch):
            try:
                self.train_step_gdp(X)
            except:
                print(self.sess.run(self.weights))
                # print(self.get_params())
                print(i)
                for j in range(self.bs):
                    qq = tf.einsum('i,ijk->jk', tf.square(self.weights[j]), tf.einsum('ijk,ilk->ijl', self.r_covs, self.r_covs))
                    print(self.sess.run(qq))
                print(i)
                raise

    def simplex_poj(self, X):
        """
        return tensor operation represents projection on simplex
        :param X:
        :return:
        """
        dual = tf.Variable(tf.truncated_normal([self.bs, 1], stddev=0.1))
        ones = tf.constant([[1.]]*self.k)

        dual_obj = -0.5*tf.reduce_sum(tf.square(tf.nn.relu(tf.matmul(dual, ones, transpose_b=True) - X)), axis=1) - \
                   dual * (tf.reduce_sum(X,axis=1, keep_dims=True) - 1) + \
                   0.5*self.k*tf.square(dual)
        proj_val = tf.assign(X, tf.nn.relu(X - tf.matmul(dual, ones, transpose_b=True)))

        u = tf.Variable(tf.ones([self.bs,1]))
        l = tf.Variable(tf.ones([self.bs,1]))
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

if __name__ == "__main__":
    means = np.vstack([[0, 10], np.ones(2) * 10, np.zeros(2)])
    covs = np.vstack([np.eye(2), np.eye(2), np.eye(2)]).reshape(3, 2, 2)

    x = None
    bs = 100
    dim = 2
    comp = 3
    with Convex_GMM(dim, comp, bs, 1e-1) as c:

        # for i in range(100):
        x = c.sample(bs, means, covs)
        c.train_gdp(x)

        c.visualize(x)
        print(c.get_params())
