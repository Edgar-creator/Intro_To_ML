import pandas as pd
import numpy as np



class MyLogisticRegression():
    def __init__(self,epsilon=1e-6,step_size=1e-4,max_steps=1000,lambd=0,fit_intercept=True):
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_steps = max_steps
        self.lambd = lambd
        self.fit_intercept = fit_intercept
        
        
        
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))
    
    def new(self,X):
        X = np.array(X)
        if self.fit_intercept is True:
            X1 = np.insert(X, 0, 1, axis=1)
            return X1
        return X
    
    def logistic_func(self,X,beta):
        return 1/(1+np.exp(np.dot(np.array(-beta),X.T)))
 
    
    def cost_func(self,X, Y,beta):
        cost=0
        regular=0
        cost=np.dot(-Y,np.log(self.logistic_func(X,beta)))-np.dot((1-Y),np.log(1-self.logistic_func(X,beta)))+ ((self.lambd/(2*X.shape[1])))*(beta**2).sum()
        return cost/X.shape[0]
     
    
    def gradient(self,X, Y,beta):

        return np.array((self.logistic_func(X,beta) -Y).T.dot(X))
    
    
    def gradient_descent(self,X, Y):
        X1 = self.new(X)
        beta= np.zeros(X1.shape[1])

        for i in range(self.max_steps):
            old_cost = self.cost_func(X1, Y,beta)
            beta -= self.step_size*self.gradient(X1, Y, beta)
            if abs(self.cost_func(X1,Y,beta) - old_cost) <= self.epsilon/5:
                print("Gradient Descent converged at %s step" % i)
                break
        self.beta = beta
                 
        if self.fit_intercept is True:
            self.intercept_=self.beta[0]
            self.coef_ = self.beta[1:]
        else:
            self.intercept_ = []
            self.coef_ = self.beta
        self.by_importance_coef_ = sorted(abs(self.coef_))
    
    def fit(self,X,Y):
        self.gradient_descent(X, Y)
        
        
    def predict(self,X):
        X = self.new(X)
        return np.where(self.logistic_func(X,self.beta)>=0.5,1,0)
   
    
    
    