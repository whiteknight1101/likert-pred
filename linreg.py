import pandas as pd
import numpy as np 
import math
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.feature_selection import SelectKBest, SelectFromModel

def read_train_data(filename):
  df = pd.read_csv(filename, header=None)
  data = df.iloc[:, 2:]
  val = df.iloc[:, 1]
  X_tmp = data.to_numpy()
  ones = np.ones((X_tmp.shape[0], 1))
  X = np.hstack([ones, X_tmp])
  Y = val.to_numpy()
  return (X,Y)

def read_test_data(filename):
  df = pd.read_csv(filename, header=None)
  samples = df.iloc[:, 0]
  samples = samples.to_numpy()
  data = df.iloc[:, 1:]
  X_tmp = data.to_numpy()
  ones = np.ones((X_tmp.shape[0], 1))
  X = np.hstack([ones, X_tmp])
  return (X,samples)

############################################
## LOGISTIC REGRESSION 
############################################

def htheta_r(x,W,r):
  # define no of classes as 9
  NCLASSES = 8

  if (r == NCLASSES):
    n = 1
    d = 1
    for i in range(NCLASSES):
      d += math.exp(W[i].dot(x))
    return (n/d)

  n = math.exp(W[r].dot(x))
  d = 1
  for i in range(NCLASSES):
    d += math.exp(W[i].dot(x))
  return n/d

def logistic_cost(x,W,r,y):
  h = htheta_r(x,W,r)
  return -(y*np.log(h)) - ((1-y)*np.log(1-h)) 

def logistic_regression(X,Y, learning_rate = 0.001, maxit = 1000, Xv = None, Yv = None, ):
  # no of samples
  m = np.shape(X)[0]  

  # no of features
  n = np.shape(X)[1]  

  # no of classes 
  c = 9

  # Create weight matrix
  w = [[0.0 for _ in range(n)] for _ in range(c)]
  w = np.matrix(w)

  wnew = [[0.0 for _ in range(n)] for _ in range(c)]
  wnew = np.matrix(wnew)

  nit = 0
  err = 0
  cost = 0

  while(nit < maxit):
    # Initialize to 0
    cost = 0

    # Iterate over all weights
    for k in range(c):
      grad = np.array([0.0 for _ in range(n)])
      # Calculate the grad and cost 
      for i in range(m):
        
        y_k = 0
        # If correct class
        if (Y[i] == k+1):
          y_k = 1
        cost += logistic_cost(X[i],w,k,y_k)
        sample_err = htheta_r(X[i],w,k) - y_k
        grad += (X[i]*sample_err)
        
        # cost[k] += sample_cost
      grad = grad/np.linalg.norm(grad)
      np.subtract(wnew[k],learning_rate*grad,wnew[k])
    
    if (nit % 50 == 0):
        print("Cost after {} iters:".format(nit), cost)

    w = np.copy(wnew)

    nit += 1

  return w

def logistic_test(wts, X_in):
    probs = [0 for w in wts]
    for i in range(len(wts)):
      probs[i] = wts[i].dot(X_in)
    return (np.argmax(probs)+1)

############################################
## GRADIENT DESCENT 
############################################

mse = []
valmse = []
its = []

def gradient_descent(X, Y, mode = 0, learning_rate = 0.001, maxit = 100000, rel_thresh = 0.00001, Xv = None, Yv = None):
    # no of samples
    m = np.shape(X)[0]  

    # no of features
    n = np.shape(X)[1]

    # initial weights
    w = [0 for _ in range(n)]
    w = np.array(w)
    
    # initial predictions and errors 
    Y_pred_init = X.dot(w)
    err_init = Y_pred_init - Y
    cost_init = (1/m)*np.sum(err_init ** 2)
    
    cost_val_prev = 1
    cost_val_curr = 1

    nit = 0
    err = 0

    while(nit < maxit):
        # Find the predicted output
        Y_pred = X.dot(w)

        # Calculate the error and cost
        err = Y_pred - Y
        cost = (1/m)*np.sum(err ** 2) 

        # If termination condition is relative cost 

        if (nit % 100 == 0):
            if (mode == 0):
                print("Train MSE:", cost)
            else:
                print("Train MSE:", cost)
                print("Validation MSE:", cost_val_prev)
            print("-----------------")  

        grad = (2/m)*(X.T).dot(err)

        # normalizing the gradient (not to be done)
        grad = grad/np.linalg.norm(grad)

        w = w - (learning_rate*grad)

        if (mode == 1):
            cost_val_curr = mse_gd(w,Xv,Yv)
            reltol = (abs(cost_val_curr - cost_val_prev))/(cost_val_prev)
            # print(reltol)
            cost_val_prev = cost_val_curr
            if (reltol <= rel_thresh):
                break

        nit += 1
    return w

def gradient_descent_ridge(X, Y, mode = 0, learning_rate = 0.001, maxit = 100000, rel_thresh = 0.00001, lmda = 5, Xv = None, Yv = None):
    # no of samples
    m = np.shape(X)[0]  

    # no of features
    n = np.shape(X)[1]  

    # initial weights
    w = [0 for _ in range(n)]
    w = np.array(w)
    
    # initial predictions and errors 
    Y_pred_init = X.dot(w)
    err_init = Y_pred_init - Y
    cost_init = ((1/m)*np.sum(err_init ** 2)) + (lmda*np.sum(w**2))
    
    cost_val_prev = 1
    cost_val_curr = 1

    nit = 0
    err = 0

    while(nit < maxit):
        w_tmp = np.append([0],w[1:])

        # Find the predicted output
        Y_pred = X.dot(w)

        # Calculate the error and cost
        err = Y_pred - Y
        cost = ((1/m)*np.sum(err ** 2)) + (lmda*np.sum(w_tmp**2))
        # print(cost)

        if (nit % 100 == 0):
            if (mode == 0):
                print("Train MSE:", cost)
            else:
                print("Train MSE:", ((1/m)*np.sum(err ** 2)))
                print("Validation MSE:", cost_val_prev)
            print("-----------------") 

        grad = ((2/m)*(X.T).dot(err)) + (2*lmda*w_tmp)
        # print(grad)

        # normalizing the gradient (not to be done)
        grad = grad/np.linalg.norm(grad)

        w = w - (learning_rate*grad)

        # If termination condition is relative cost 
        if (mode == 1):
            cost_val_curr = mse_gd(w,Xv,Yv) + (lmda*np.sum(w_tmp**2))
            reltol = (abs(cost_val_curr - cost_val_prev))/(cost_val_prev)
            # print(reltol)
            cost_val_prev = cost_val_curr
            if (reltol <= rel_thresh):
                break

        nit += 1
    
    return w

def mse_gd(w, X, Y):
  m = np.shape(X)[0]
  error = Y - X.dot(w)
  cost = (1/m)*np.sum(error ** 2)
  return cost

def mae_gd(w, X, Y):
    m = np.shape(X)[0]
    error = Y - X.dot(w)
    error = np.absolute(error)
    cost = (1/m)*np.sum(error)
    return cost

def mse_gd_pred(Y_pred, Y):
  m = len(Y)
  error = Y - Y_pred
  cost = (1/m)*np.sum(error ** 2)
  return cost

def mae_gd_pred(Y_pred, Y):
    m = np.shape(Y)[0]
    error = Y - Y_pred
    error = np.absolute(error)
    cost = (1/m)*np.sum(error)
    return cost

############################################
## USING SCIKIT LEARN
############################################

def sci_linear_regr(X, Y):
    reg = LinearRegression().fit(X, Y)
    return reg

def sci_select_best(X,Y,nfeatures):
    X_tmp = X[:, 1:]
    selector = SelectKBest(k=nfeatures)
    selector.fit_transform(X_tmp, Y)
    features_cols = selector.get_support(indices=True)
    features_cols = np.append([0], features_cols)
    X_new = X[:, features_cols] 
    return (X_new, Y, features_cols)

def sci_model_best(X,Y,nfeatures):
    X_tmp = X[:, 1:]
    est = Ridge(alpha=5)
    selector = SelectFromModel(estimator=est, max_features=nfeatures)
    selector.fit_transform(X_tmp, Y)
    features_cols = selector.get_support(indices=True)
    features_cols = np.append([0], features_cols)
    X_new = X[:, features_cols]
    return (X_new, Y, features_cols)


##################################################
### ONE VS ALL LOGISTIC REGRESSION
##################################################

##################################################
### ONE VS ALL LOGISTIC REGRESSION
##################################################

def logistic_test_one_vs_all(wts, x):
    probs = [0 for w in wts]
    for i in range(len(wts)):
      probs[i] = sigmoid(x,wts,i)
    return (np.argmax(probs)+1)

def sigmoid(x,W,r):
  # define no of classes as 9
  
  n = math.exp(W[r].dot(x))
  d = 1 + math.exp(W[r].dot(x))

  return n/d

def logistic_cost_one_vs_all(x,W,r,y):
  h = sigmoid(x,W,r)
  return -((y*np.log(h)) + ((1-y)*np.log(1-h)))

def logistic_one_vs_all(X,Y, learning_rate = 0.001, maxit = 1000, Xv = None, Yv = None, ):
  # no of samples
  m = np.shape(X)[0]  

  # no of features
  n = np.shape(X)[1]  

  # no of classes 
  c = 9

  # Create weight matrix
  w = [[0.0 for _ in range(n)] for _ in range(c)]
  w = np.matrix(w)

  wnew = [[0.0 for _ in range(n)] for _ in range(c)]
  wnew = np.matrix(wnew)

  nit = 0
  err = 0
  cost = 0

  while(nit < maxit):
    # Initialize to 0
    for k in range(c):
        cost = 0
        grad = np.array([0.0 for _ in range(n)])
        for i in range(m):
            y_k = 0
            # If correct class
            if (Y[i] == k+1):
                y_k = 1
            cost += logistic_cost_one_vs_all(X[i],w,k,y_k)
            sample_err = sigmoid(X[i],w,k) - y_k
            grad += (X[i]*sample_err)
        grad = grad/np.linalg.norm(grad)
        np.subtract(wnew[k],learning_rate*grad,wnew[k])
        # print(cost)

        # if (nit % 50 == 0):
        #     print("Cost:", cost)

    if (nit % 50 == 0):
        correct = 0
        total = 0
        for j in range(len(Xv)):
            pred = logistic_test_one_vs_all(wnew, Xv[j])
            actual = Yv[j]
            if (pred == actual):
                correct += 1
            total += 1
            # print(logistic_test(w_train, X_train[j]), Y_train[j])
        acc = correct/total
        print("After {} iters, VA:".format(nit), acc)
    
    w = np.copy(wnew)

    nit += 1
        
  return w