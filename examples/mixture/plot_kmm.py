import numpy as np

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from sklearn.mixture import GMM, KentMixtureModel
from sklearn.mixture.kmm import to_spherical_coords, to_cartesian_coords

if __name__ == "__main__":

    np.random.seed(123456)

    # ------------------------------------

    #X = np.array([[0,0.1,1], [0,0.1,0.5], [0,-0.1,0.5], [0.5,0,0.5], [-0.8,0,0.2]])
    #X_raw = np.array([[0,0.1,1], [0,0.1,0.9], [0,-0.1,0.9], [0.5,0,0.9], [-0.5,0,0.9]])
    #X_raw = np.array([[0,0.1,1],])
    #X_raw = np.array([[0,0.1,1], [0,1,-0.1]])
    
    #X_raw = np.array([[0,0.1,1], [0,1,-0.1], [1,-1,0.1]])
    #X_raw = np.array([[0,1,1], [0,1,-0.5], [1,-0.5,0.5]])
    #X_raw = np.array([[0,0.1,1], [0,0.5,1.], [1,-0.5,0.5]])
    #X_raw = np.array([[0.4,0.,0.5], [0,0,1.0], [-0.4,0,0.5], ])
    
    # ground truth centers
    X_raw = np.array([[1.0,0,0], [0.5,0.,0.5], [0,0,1.0]])
    k = X_raw.shape[0]
    
    samples = 50
    std = [0.6, 0.05]
    #std = [0.2, 0.05]
    
    responsibilities_true = np.zeros( (samples*X_raw.shape[0], X_raw.shape[0]) )
    clustering_true = np.zeros( (samples*X_raw.shape[0], ) )
    
    # generate data
    X = np.zeros( (samples*X_raw.shape[0], X_raw.shape[1]) )
    row = 0
    for i in range(X_raw.shape[0]):
      X_raw[i,:] /= np.linalg.norm(X_raw[i,:])
      gamma_raw = to_spherical_coords(X_raw[i, :])
     
      for _ in range(samples):
        gamma = gamma_raw + np.random.randn(2)*std
        X[row,:] = to_cartesian_coords(gamma)
        responsibilities_true[row, :] = [ 1. if i==j else 0. for j in range(X_raw.shape[0]) ]
        clustering_true[row] = i
        row += 1

    # ------------------------------------
    # fit Kent mixture model

    iterations = 500
    initializations = 10

    klf = KentMixtureModel(n_components=k, n_iter=iterations, n_init=initializations)
    klf.fit(X)
    kmm_lpr, kmm_responsibilities = klf.score_samples(X)

    # ------------------------------------
    # fit GMM
    glf = GMM(n_components=k, n_iter=iterations, n_init=initializations, covariance_type='full')
    #glf = GMM(n_components=k, n_iter=iterations, n_init=initializations, covariance_type='diag')
    glf.fit(X)
    gmm_lpr, gmm_responsibilities = glf.score_samples(X)

    # ------------------------------------
    # Print some info
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print "CENTERS"
    print [X_raw[i,:] / np.linalg.norm(X_raw[i,:]) for i in range(X_raw.shape[0])]

    print "true clustering"
    print clustering_true

    print "DATA"
    print X

    print "KMM: responsibilities"
    print kmm_responsibilities.T
    print "kappa: %s,\nbeta: %s,\nG:\n %s\nS:\n:%s" % tuple(map(str,(klf.kappa_, klf.beta_, klf.G_, klf.S_)))
    print "weights: %s" % str(klf.weights_)

    #kmm_clustering = np.argmax(kmm_responsibilities, 1)
    ## find mapping
    #kmm_mapping=[0.]*X_raw.shape[0]
    #for i in range(X_raw.shape[0]):
        #count = np.bincount(kmm_clustering[i*samples:(i+1)*samples])
        #kmm_mapping[i] = np.argmax(count)
    #kmm_clustering = [ kmm_mapping[i] for i in kmm_clustering ]
    #print "KMM clustering"
    #print kmm_mapping
    #print kmm_clustering

    print "GMM: responsibilities"
    print gmm_responsibilities.T
    print "Mean: %s,\nCov:\n%s" % tuple(map(str, (glf.means_, glf.covars_)))

    #gmm_clustering = np.argmax(gmm_responsibilities, 1)
    ## find mapping
    #gmm_mapping=[0.]*X_raw.shape[0]
    #for i in range(X_raw.shape[0]):
        #count = np.bincount(gmm_clustering[i*samples:(i+1)*samples])
        #gmm_mapping[i] = np.argmax(count)
    #gmm_clustering = [ gmm_mapping[i] for i in gmm_clustering ]
    #print "GMM clustering"
    #print gmm_mapping
    #print gmm_clustering

    # ------------------------------------
    # visualize
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
            

    colors=['r','b', 'y', 'k', 'c',  'g', 'white', 'm', 'turquoise', 'darkgreen', ]

    n_components = X_raw.shape[0]
    for name, clf, lpr, responsibilities, means in [ 
        ("Truth", None, np.ones(X.shape[0]), responsibilities_true, X_raw),
        ("Kent", klf, kmm_lpr, kmm_responsibilities, klf.G_[:, :, 0]),
        ("Gaussian", glf, gmm_lpr, gmm_responsibilities, glf.means_) ]:

      print "==============================="
      print name

      fig = plt.figure(figsize=(10,10))
      ax = fig.add_subplot(1,1,1, projection='3d')
      plt.title(name)

      #print "Log-likelihood"
      #print np.sum(lpr)

      X_lik = np.exp(lpr)
      #print "Likelihoods: "
      #print X_lik
      X_lik /= np.max(X_lik)
      print "Likelihoods (normalized): "
      print X_lik

      for k in range(n_components):
        G = means[k]

        print "Mean vectors"
        print G
      
        # cluster mean
        ax.scatter(G[0], G[1], G[2], c="k", s = 100.0, alpha=1.0)
        a = Arrow3D([0, G[0]], [0, G[1]], [0, G[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        ax.text(G[0], G[1], G[2]+0.1, "C%d" % k, color="black")
        ax.add_artist(a)

        # true mean
        ax.scatter(X_raw[k,0], X_raw[k,1], X_raw[k,2], c="w", s = 100.0, alpha=1.0)
        a = Arrow3D([0, X_raw[k,0]], [0, X_raw[k,1]], [0, X_raw[k,2]], mutation_scale=20, lw=0.5, arrowstyle="-|>", color="k")
        ax.add_artist(a)

        ## major axis
        if name == "Kent":
          pass
          #G = clf.G_[k]
          #ax.scatter(G[0,1:2], G[1,1:2], G[2,1:2], c="b", s = 100.0, alpha=1.0)
          #a = Arrow3D([0, G[0,1:2]], [0, G[1,1:2]], [0, G[2,1:2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
          #ax.add_artist(a)
          ## minor axis
          #ax.scatter(G[0,2:3], G[1,2:3], G[2,2:3], c="y", s = 100.0, alpha=1.0)
          #a = Arrow3D([0, G[0,2:3]], [0, G[1,2:3]], [0, G[2,2:3]], mutation_scale=20, lw=1, arrowstyle="-|>", color="y")
          #ax.add_artist(a)

      for i in range(X.shape[0]):
          k = np.argmax(responsibilities[i,:])
          #if responsibilities[i,k] < np.max(responsibilities[i,:]):
          #  continue
          lik = np.max([0.6, np.min([1.0, X_lik[i]])])
          
          ax.scatter(X[i:i+1,0], X[i:i+1,1], X[i:i+1,2], c=colors[k], s = 50.0, alpha=lik)
          a = Arrow3D([0, X[i:i+1,0]], [0, X[i:i+1,1]], [0, X[i:i+1,2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="gray")
          ax.add_artist(a)
          ax.text(X[i:i+1,0], X[i:i+1,1], X[i:i+1,2]+0.1, "%d" % i, color="gray")

      ax.set_xlim3d(-1,1)
      ax.set_ylim3d(-1,1)
      ax.set_zlim3d(-1,1)

      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel("z")

    plt.show()
