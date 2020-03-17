import numpy as np


'''
Receives NN weights, bias, and initial entries matrix 9 x m
Calculates and returns the result 9 x m vector
'''
def forwardPropagation(W, b, X):
  assert isinstance(W, np.ndarray)
  # assert len(W.shape) == 3
  assert isinstance(b, np.ndarray)
  # assert len(b.shape) == 3
  assert W[0].shape[0] == b[0].shape[0]
  assert isinstance(X, np.ndarray)
  assert len(X.shape) == 2
  assert(X.shape[0] == 9)

  Z = [None]
  A = [X]
  L = len(W)

  for i in range(L):
    Z.append(np.dot(W[i], A[i]) + b[i])
    A.append(sigmoid(Z[i + 1]))

  return A[L]


'''
Receives expected (Y) and calculated result (Yhat)
(both are numpy arrays 9 x m)
Calculates and returns cost
'''
def costFunction(Y, Yhat):
  assert isinstance(Y, np.ndarray)
  assert len(Y.shape) == 2
  assert isinstance(Yhat, np.ndarray)
  assert Y.shape == Yhat.shape

  m = Y.shape[1]
  cost = Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat)
  cost = np.sum(cost)
  cost = cost * -1 / m
  return cost


'''
Receives weights, bias, and initial entries
Calculates and returns derivatives for each weight and bias
'''
def calcGradients(W, b, X, Y, epsilon = 1e-3):
  theta = reshapeInTheta(W, b)
  dTheta = np.array([])
  WCopy = np.copy(W)

  for i in range(len(theta)):
    theta1 = np.copy(theta)
    theta1[i] -= epsilon
    (W1, b1) = reshapeFromTheta(theta1, WCopy)
    AL1 = forwardPropagation(W1, b1, X)
    cost1 = costFunction(Y, AL1)

    theta2 = np.copy(theta)
    theta2[i] += epsilon
    (W2, b2) = reshapeFromTheta(theta2, WCopy)
    AL2 = forwardPropagation(W2, b2, X)
    cost2 = costFunction(Y, AL2)

    d = (cost2 - cost1) / (2 * epsilon)

    dTheta = np.append(dTheta, d)

  dTheta = np.array(dTheta).reshape(dTheta.size, 1)

  (dW, db) = reshapeFromTheta(dTheta, WCopy)

  return (dW, db)


'''
Receives NN weights and bias
Reshapes them into a vector
'''
def reshapeInTheta(W, b):
  assert isinstance(W, np.ndarray)
  # assert len(W.shape) == 3
  assert isinstance(b, np.ndarray)
  # assert len(b.shape) == 3

  L = W.shape[0]
  theta = np.array([])

  for i in range(L):
    theta = np.append(theta, W[i])
    theta = np.append(theta, b[i])

  theta = theta.reshape(theta.size, 1)

  return theta


'''
Receives theta and W
Returns array of W and b
'''
def reshapeFromTheta(theta, WCopy):
  assert isinstance(theta, np.ndarray)
  assert theta.shape[1] == 1

  assert isinstance(WCopy, np.ndarray)
  # assert len(WCopy.shape) == 3  

  W = []
  b = []
  startIndex = 0

  for i in range(len(WCopy)):
    wSize = WCopy[i].size
    endIndex = startIndex + wSize
    W.append(theta[startIndex:endIndex])
    W[i] = W[i].reshape(WCopy[i].shape)

    bSize = WCopy[i].shape[0]
    startIndex = endIndex
    endIndex = startIndex + bSize
    b.append(theta[startIndex:endIndex])

  W = np.array(W)
  b = np.array(b)

  return (W, b)


'''
Receives weights, their derivatives, and learning rate.
Returns updated weights
'''
def updateWeights(W, dW, b, db, alpha):
  assert isinstance(W, np.ndarray)
  assert isinstance(dW, np.ndarray)
  assert W.shape == dW.shape

  assert isinstance(b, np.ndarray)
  assert isinstance(db, np.ndarray)
  assert b.shape == db.shape

  for i in range(len(W)):
    W[i] = W[i] - alpha * dW[i]
    b[i] = b[i] - alpha * db[i]


'''
Receives model structure:
an array of NN layers with number of nodes in each layer
Generates and returns random initial weights and bias
'''
def initializeWeights(n):
  assert isinstance(n, list)

  L = len(n)
  W = []
  b = []

  for i in range(1, L):
    layerWeights = np.random.randn(n[i], n[i - 1]) * np.sqrt(2. / n[i - 1])
    layerBias = np.zeros((n[i], 1))

    W.append(layerWeights)
    b.append(layerBias)
  
  W = np.array(W)
  b = np.array(b)

  return (W, b)



'''
Receives weights, bias and initial position
Returns best movement
'''
def predict(W, b, x):
  assert isinstance(W, np.ndarray)
  assert isinstance(b, np.ndarray)
  assert isinstance(x, np.ndarray)
  assert x.shape == (9, 1)

  a = forwardPropagation(W, b, x)
  maxIndex = a.argmax()

  if x[maxIndex] == 0:
    y = np.zeros((9, 1))
    y[maxIndex] = 1

    return y

  a[maxIndex] = 0

  maxIndex = a.argmax()

  if x[maxIndex] == 0:
    y = np.zeros((9, 1))
    y[maxIndex] = 1

    return y

  a[maxIndex] = 0

  maxIndex = a.argmax()

  y = np.zeros((9, 1))
  y[maxIndex] = 1

  return y

'''
Receives matrix of values.
Returns sigmoid activations for those values.
'''
def sigmoid(Z):
  assert isinstance(Z, np.ndarray)
  assert len(Z.shape) == 2

  return 1 / (1 + np.exp(-Z))
