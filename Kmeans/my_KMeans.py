import pandas as pd
import numpy as np
from pdb import set_trace

class my_KMeans:

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=1e-4):
        # Initializing the K-Means instance
        # The number of clusters
        # Initialization method can be "k-means++" or "random"
        # We stop when either the number of iterations exceeds max_iter or the change in inertia is less than tol.
        # We repeat the process n_init times and keep the best run (cluster_centers_, inertia_) with the lowest inertia_.
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol

        # Define cluster labels (0 to n_clusters-1)
        self.classes_ = range(n_clusters)
        # Initialize cluster centers
        self.cluster_centers_ = None
        # Inertia represents the sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None

    def dist(self, a, b):
        # Calculating the Euclidean distance between data points a and b
        return np.sum((np.array(a) - np.array(b))**2)**(0.5)

    def initiate(self, X):
        # Initiating cluster centers
        # Input X is a numpy array
        # Output cluster_centers (list)

        if self.init == "random":
            # Initialize cluster centers randomly
            cluster_centers = [X[i] for i in np.random.choice(len(X), self.n_clusters, replace=False)]

        elif self.init == "k-means++":
            # Initialize cluster centers using k-means++
            cluster_centers = [X[np.random.choice(len(X))]]

            while len(cluster_centers) < self.n_clusters:
                min_dists = []
                for x in X:
                    dist = min([self.dist(x, center) for center in cluster_centers])**2
                    min_dists.append(dist)
                total_dist = sum(min_dists)
                probs = [dist / total_dist for dist in min_dists]
                next_center = X[np.random.choice(len(X), p=probs)]
                cluster_centers.append(next_center)

        else:
            raise Exception("Unknown value of self.init.")
        return cluster_centers

    def fit_once(self, X):
        # Fitting the model for a single run
        # Input X is a numpy array
        # Output: cluster_centers (list), inertia

        # Initialize cluster centers
        cluster_centers = self.initiate(X)
        last_inertia = None
        # Iterate
        for i in range(self.max_iter+1):
            # Assigning each training data point to its nearest cluster center
            clusters = [[] for i in range(self.n_clusters)]
            inertia = 0
            for x in X:
                # Calculating distances between x and each cluster center
                dists = [self.dist(x, center) for center in cluster_centers]
                # Calculating inertia
                inertia += min(dists)**2
                # Finding the cluster that x belongs to
                cluster_id = dists.index(min(dists))
                # Adding x to that cluster
                clusters[cluster_id].append(x)

            if (last_inertia and last_inertia - inertia < self.tol) or i == self.max_iter:
                break
            # Updating cluster centers
            new_centers = [np.mean(clusters[i], axis=0) for i in range(self.n_clusters)]
            cluster_centers = new_centers
            last_inertia = inertia

        return cluster_centers, inertia

    def fit(self, X):
        # Fitting the model with the given data
        # X: pd.DataFrame, independent variables, float
        # Repeating the process self.n_init times and keeping the best run
        # (self.cluster_centers_, self.inertia_) with the lowest self.inertia_.
        X_feature = X.to_numpy()
        for i in range(self.n_init):
            cluster_centers, inertia = self.fit_once(X_feature)
            if self.inertia_ is None or inertia < self.inertia_:
                self.inertia_ = inertia
                self.cluster_centers_ = cluster_centers
        return

    def transform(self, X):
        # Transforming data to cluster-distance space
        # X: pd.DataFrame, independent variables, float
        # Returning dists = list of [distance to centroid 1, distance to centroid 2, ...]
        dists = [[self.dist(x, centroid) for centroid in self.cluster_centers_] for x in X.to_numpy()]
        return dists

    def predict(self, X):
        # Predicting cluster labels for the given data
        # X: pd.DataFrame, independent variables, float
        # Returning predictions: list
        predictions = [np.argmin(dist) for dist in self.transform(X)]
        return predictions

    def fit_predict(self, X):
        # Fitting the model and predicting cluster labels
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        # Fitting the model and transforming data to cluster-distance space
        self.fit(X)
        return self.transform(X)
