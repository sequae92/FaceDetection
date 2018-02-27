##  Defining the EM module
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize
from dataLoading import *

'''
    Iterations - Max number of iterations the model is going to run till
    Clusters - Max number of clusters
    Data - Will be the feature vector of the dataset (class-specific) -- this should be SCALED
    Parameters - Will specify the number of parameters ( Gaussian - 2)
'''
class ExpectationMaximisation(object):

    def __init__(self, m, c, d, n): # initialising all the parameters needed for the functionality of EM algorithm
        self.maxiters = m
        self.clusters = c
        self.data = d
        self.model = [None]*c
        self.cthreshold = 0.00001
        self.x = self.data.shape[0]
        self.y = self.data.shape[1]
        self.likelihood = np.zeros((self.x, self.clusters))
        self.r = np.zeros((self.x, self.clusters))

    def expectation(self,data):
        for k in range(self.clusters):
            #print("Mean ",self.para[k][0])
            # print("Covariance",self.para[k][1])
            m = multivariate_normal(self.para[k][0],self.para[k][1])
            self.model[k] = m
            self.likelihood[:,k] = np.multiply(self.like[k],self.model[k].pdf(self.data))

        self.r = normalize(self.likelihood, norm='l1', axis=1)

    def maximisation(self,data):
        ## The M step starts here
        for k in range(self.clusters):
            self.like[k] = np.sum(self.r,axis=0)[k] / np.sum(self.r)  # this is the new prior
            wr = self.r[:, k]
            self.para[k][0] = np.average(data, weights=wr, axis=0)
            self.para[k][1] = np.cov(data, aweights= wr, rowvar = False)
        # for ends

    def calc_L(self,data):
        # find the data log likelihood
        L = 0.0
        prob = np.zeros((1000,2))
        for k in range(self.clusters):
            prob[:,k] = np.multiply(self.like[k],self.model[k].pdf(self.data))
        prob = np.sum(prob,axis=1)
        p = np.log(prob)
        L = np.sum(p)
        return L

    '''
    For initiliasing all the variables which will be used for calculating E- step and M-step
    '''

    def initialization(self):
        print("starting init")
        self.like = np.ones(self.clusters) / self.clusters  # setting the prior initially to 1/classes
        mean = np.zeros((self.clusters, self.y)) # initilaises the mean to a random row in the dataset

        for i in range(self.clusters):
            idx = np.random.randint(self.x, size=1)
            mean[i,:] = self.data[idx, :]

        cov = np.cov(self.data, rowvar=False)  # this finds the covariance of the data
        print("reached here")
        # Initialising the parameters
        # The parameters will be mean and covariance

        self.para = {}
        for i in range(self.clusters):
            self.para[i] = [mean[i,:], cov]
            self.model[i] = multivariate_normal(mean[i,:], cov)

        print("models generated")
        self.lold = self.calc_L(self.data)
    '''
        For defining the EM algorithm
    '''
    def eMaximisation(self):
        self.initialization()

        print("=======================================================================")

        iters = 0
        while (iters <= self.maxiters):
            iters = iters + 1

            self.expectation(self.data)
            self.maximisation(self.data)
            Lnew = self.calc_L(self.data)
            print(" Lnew : Lold ",Lnew,self.lold)
            if (Lnew > self.lold):
                self.lold = Lnew
            else:
                break
            print("iteration = ", iters)
        return [self.model,self.para,self.like]

    '''
    Loads data into feature vectors
    '''
if __name__ == '__main__':
    classes = [0,1]
    result = [None]*2 # Here 2 shows the face and non-face images
    em = [None]*2

    for i in classes:
        if not i:
            print("For the Non-Face images.......")
        else:
            print("For the Face images......")

        # data loading
        data1 = loadData(i)
        print(data1.shape)

        # dataoperations
        dataNew = dataOperations(data1)
        print(dataNew.shape)

        # Creating EM object with the parameters
        # Header parameters - maxiters,clusters,data,parameters
        em[i] = ExpectationMaximisation(500, 2, dataNew, 2) # Here the clusters is 2

     # while the EM does not converge
        result[i] = em[i].eMaximisation()
        print("Model and the corresponding parameters:", result[i])
        print("==========================================================")