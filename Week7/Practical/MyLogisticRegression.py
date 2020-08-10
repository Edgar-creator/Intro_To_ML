import pandas as pd
import numpy as np



class MyLogisticRegression():
    beta = np.array([])
    def __init__(self,epsilon=1e-6,step_size=1e-4,max_steps=1000,lambd=0,fit_intercept=False):
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_steps = max_steps
        self.lambd = lambd
        self.fit_intercept = fit_intercept
        
        
        
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))
    
    def new(self,X):
        if self.fit_intercept is True and "for_intercept" not in X.columns:
            x=X.copy()
            x["for_intercept"]=1
            x = x[[x.columns[-1]] + list(x.columns[:-1])]
            return x
        return X
    
    def logistic_func(self,X,beta):
        X = self.new(X)
        return 1/(1+np.exp(np.dot(np.array(-beta),X.T)))
 
    
    def cost_func(self,X, Y,beta):
        cost=0
        regular=0
        X1 = self.new(X)
        if  X1.shape[1]==X.shape[1]:
            cost=np.dot(-Y,np.log(self.logistic_func(X,beta)))-np.dot((1-Y),np.log(1-self.logistic_func(X,beta)))+ ((self.lambd/(2*X.shape[1])))*(beta**2).sum()
            return cost/X.shape[0]
        else:
            cost=np.dot(-Y,np.log(self.logistic_func(X1,beta)))-np.dot((1-Y),np.log(1-self.logistic_func(X1,beta)))
            regular = ((self.lambd/(2*X.shape[1])))*(beta[1:]**2).sum()
            return cost/X1.shape[0] + regular/(X1.shape[0]-1)
    
    def gradient(self,X, Y,beta):

        return np.array((self.logistic_func(X,beta) -Y).T.dot(X))
    
    
    def gradient_descent(self,X, Y):
        X1 = self.new(X)
        beta= np.zeros(X1.shape[1])

        for i in range(self.max_steps):
            if X1.shape[1]==X.shape[1]:
                old_cost = self.cost_func(X, Y,beta)
            
                beta -= self.step_size*self.gradient(X, Y,beta)
                if abs(self.cost_func(X,Y,beta) - old_cost) <= self.epsilon:
                    print("Gradient Descent converged at %s step" % i)
                    break
            else:
                old_cost = self.cost_func(X, Y,beta)
                beta -= self.step_size*self.gradient(X1, Y, beta)
                if abs(self.cost_func(X1,Y,beta) - old_cost) <= self.epsilon/3:
                    print("Gradient Descent converged at %s step" % i)
                    break
        
        self.beta=beta

                
     