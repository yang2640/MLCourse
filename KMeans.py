#created by yang zhou on Jan. 29

import numpy as np
import random

def L1_dist(x, y):
	"""
	x : one long sample vector
	y : array of cluster bases
	"""
	return np.sum(abs(x-y), axis=1)

def KMeans(N, X, tol, max_iter, metric, centers):
	"""
	N : number of clusters
	X : samples in numpy array, sample_num * feat_num
	tol : cost < tol, while loop will break
	max_iter : maximun number of iterations kmeans will run
	metric : distance metric, support L1-distance here
	centers : use determined init cluster centers
	"""
	if metric == "l1":
		sim = L1_dist;
	
	sample_num = X.shape[0]
	feat_dim = X.shape[1]
	#init random N random cluster centers
	if len(centers) == 0:
		randint = random.sample(range(sample_num), N)
		centers = X[randint,:]
	iter = 0
	while True:
		#compute cluster assignments
		cluster_assign = np.array([sim(x, centers).argsort()[0] for x in X])

		#re-compute the cluster centers
		for n in xrange(N):
			centers[n] = np.mean(X[n==cluster_assign], axis=0)

		#compute the cost funtion
		A = np.zeros(X.shape)
		for n in xrange(sample_num):
			A[n] = centers[cluster_assign[n]]

		cost = np.mean(np.sqrt(np.sum((X-A)**2,axis=1)))

		iter = iter + 1 
		if (cost < tol or iter > max_iter):
			break

	return cluster_assign

if __name__ ==  "__main__":
	X = np.loadtxt("feat.txt")
	N = 3
	centers = X[[0,3,6],:]
	#if want to rondomly init centers, set centers = []
        clusters = KMeans(N=N, X=X, tol=0.1, max_iter=20, metric="l1", centers=centers)
	for n in xrange(N):
		print "cluster %d:"%(n+1),np.where(n==clusters)[0]+1 


