"""
The example shows the usage of the KentMixture.

It shows the performance of the Kent mixture model in comparison
to a Gaussian mixture model on synthetic data.
"""

import numpy as np

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from sklearn.mixture import GaussianMixture, KentMixture
from sklearn.mixture._kent_mixture import to_spherical_coords, to_cartesian_coords

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


if __name__ == "__main__":
    np.random.seed(123456)

    colors=['r','b', 'y', 'k', 'c',  'g', 'white', 'm', 'turquoise', 'darkgreen', ]

    # ------------------------------------
    # ground truth centers
    
    centers = np.array([[1.0,0,0], [0.,1.,0.], [-1,0,0.]])
    n_components = centers.shape[0]
    
    # no samples per center
    samples = 60
    # standard deviation in angle per center
    stds = np.array([[0.3, 0.75], [0.1,0.5], [0.7,0.3]])
    
    responsibilities_true = np.zeros( (samples*centers.shape[0], centers.shape[0]) )
    clustering_true = np.zeros( (samples*centers.shape[0], ) )
    
    # generate data
    X = np.zeros( (samples*centers.shape[0], centers.shape[1]) )
    row = 0
    for i in range(centers.shape[0]):
      # normalize center
      centers[i,:] /= np.linalg.norm(centers[i,:])
      # convert to spherical coordinates
      gamma_raw = to_spherical_coords(centers[i, :])
     
      for _ in range(samples):
        gamma = gamma_raw + np.random.randn(2)*stds[i]
        X[row,:] = to_cartesian_coords(gamma)
        responsibilities_true[row, :] = [ 1. if i==j else 0. for j in range(centers.shape[0]) ]
        clustering_true[row] = i
        row += 1

    iterations = 500
    initializations = 10

    # ------------------------------------
    # fit Kent mixture model
    klf = KentMixture(n_components=n_components, n_iter=iterations, n_init=initializations)
    klf.fit(X)
    kmm_lpr, kmm_responsibilities = klf.score_samples(X)

    # ------------------------------------
    # fit GaussianMixture
    glf = GaussianMixture(n_components=n_components, n_iter=iterations, n_init=initializations, covariance_type='full')
    glf.fit(X)
    gmm_lpr, gmm_responsibilities = glf.score_samples(X)

    # ------------------------------------
    # Print some info
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print ("CENTERS")
    print ([centers[i,:] / np.linalg.norm(centers[i,:]) for i in range(centers.shape[0])])

    print ( "KMM: responsibilities" )
    print ( kmm_responsibilities.T )
    print ( "kappa: %s,\nbeta: %s,\nG:\n %s\nS:\n%s" % tuple(map(str,(klf.kappa_, klf.beta_, klf.G_, klf.S_))) )
    print ( "weights: %s" % str(klf.weights_) )

    print ( "GaussianMixture: responsibilities" )
    print ( gmm_responsibilities.T )
    print ( "Mean: %s,\nCov:\n%s" % tuple(map(str, (glf.means_, glf.covars_))) )

    # ------------------------------------
    # visualize
    for name, clf, lpr, responsibilities, means in [ 
        ("Truth", None, np.ones(X.shape[0]), responsibilities_true, centers),
        ("Kent", klf, kmm_lpr, kmm_responsibilities, klf.G_[:, :, 0]),
        ("Gaussian", glf, gmm_lpr, gmm_responsibilities, glf.means_) ]:

      print ( "===============================" )
      print ( name )

      fig = plt.figure(figsize=(10,10))
      ax = fig.add_subplot(1,1,1, projection='3d')
      plt.title(name)

      print ( "Log-likelihood" )
      print ( np.sum(lpr) )

      X_lik = np.exp(lpr)
      X_lik /= np.max(X_lik)
      print ( "Likelihoods (normalized): " )
      print ( X_lik )

      cluster_mapping = {}
      for k in range(n_components):
        G = means[k]

        print ( "Mean vectors" )
        print ( G )
      
        # find matching real cluster
        k_true = np.argmin(1 - np.dot(centers, G))
        cluster_mapping[k] = k_true
      
        # cluster mean
        ax.scatter(G[0], G[1], G[2], c="k", s = 100.0, alpha=1.0)
        a = Arrow3D([0, G[0]], [0, G[1]], [0, G[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        ax.text(G[0], G[1], G[2]+0.1, "C%d" % k, color="black")
        ax.add_artist(a)

        # true mean
        ax.scatter(centers[k,0], centers[k,1], centers[k,2], c="w", s = 100.0, alpha=1.0)
        a = Arrow3D([0, centers[k,0]], [0, centers[k,1]], [0, centers[k,2]], mutation_scale=20, lw=0.5, arrowstyle="-|>", color="k")
        ax.add_artist(a)

      misclassified = 0
      for i in range(X.shape[0]):
          j = np.argmax(responsibilities[i,:])
          lik = np.max([0.4, np.min([1.0, X_lik[i]])])
          
          if cluster_mapping[j] != clustering_true[i]:
            misclassified += 1
          
          ax.scatter(X[i:i+1,0], X[i:i+1,1], X[i:i+1,2], c=colors[cluster_mapping[j] ], s = 50.0, alpha=lik)
          a = Arrow3D([0, X[i:i+1,0]], [0, X[i:i+1,1]], [0, X[i:i+1,2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="gray")
          ax.add_artist(a)
          ax.text(X[i:i+1,0], X[i:i+1,1], X[i:i+1,2]+0.1, "%d" % i, color="gray")
      print ( "Misclassified: %d " % misclassified )

      ax.set_xlim3d(-1,1)
      ax.set_ylim3d(-1,1)
      ax.set_zlim3d(-1,1)

      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel("z")

    plt.show()
