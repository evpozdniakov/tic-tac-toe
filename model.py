import numpy as np



'''
Receives NN weights, bias, and initial entries matrix 9 x m
Calculates and returns A for each layer
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
    # print('W[i].shape')
    # print(W[i].shape)
    # print('A[i].shape')
    # print(A[i].shape)
    # exit()
    Z.append(np.dot(W[i], A[i]) + b[i])
    A.append(sigmoid(Z[i + 1]))

  # m = X.shape[1]
  # ALexp = np.exp(A[L])
  # ALexpSum = np.sum(ALexp, axis = 0).reshape(1, m)
  # A[L] = ALexp / ALexpSum

  return (A[L], A)



'''
Receives Y and Yhat
(both are numpy arrays 9 x m)
Calculates classic cost function and returns it
'''
def costFunction(Y, Yhat):
  assert isinstance(Y, np.ndarray)
  assert len(Y.shape) == 2
  assert isinstance(Yhat, np.ndarray)
  assert Y.shape[0] == 9
  assert Y.shape == Yhat.shape

  m = Y.shape[1]

  cosmMatrix = -1 * (Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))

  cost = np.sum(cosmMatrix) / m

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
    (aL1, _) = forwardPropagation(W1, b1, X)
    cost1 = costFunction(Y, aL1)

    theta2 = np.copy(theta)
    theta2[i] += epsilon
    (W2, b2) = reshapeFromTheta(theta2, WCopy)
    (aL2, _) = forwardPropagation(W2, b2, X)
    cost2 = costFunction(Y, aL2)

    d = (cost2 - cost1) / (2 * epsilon)

    dTheta = np.append(dTheta, d)

  dTheta = np.array(dTheta).reshape(dTheta.size, 1)

  (dW, db) = reshapeFromTheta(dTheta, WCopy)

  return (dW, db)



'''
Receives layers, weights, bias, initial entries, and Y
Calculates and returns derivatives for each weight and bias
using back-prop
'''
def backPropagation(n, W, b, X, Y):
  L = len(n) - 1 # L=1 for n=[9,9] as first layer is "zero" layer
  assert len(W) == L

  m = Y.shape[1]

  # calculate A for all layers
  (_, A) = forwardPropagation(W, b, X)


  dW = []
  db = []

  for l in range(L, 0, -1): # l=2,1 for L=2, n=[9,18,9]
    if l == L:
      dZl = (A[L] - Y)
    else:
      dZl = A[l] * (1 - A[l]) * da

    da = np.dot(W[l - 1].T, dZl)

    dWl = np.dot(dZl, A[l - 1].T)
    dWl = dWl / m
    dW.insert(0, dWl)

    dbl = dZl.sum(axis = 1).reshape(b[l - 1].shape) / m
    db.insert(0, dbl)

  dW = np.array(dW)
  db = np.array(db)

  return (dW, db, A)



'''
Receives layers, weights, bias, initial entries and Y
Compares detivatives calculated with calcGradients and backPropagation
(if the difference between them is less than epsilon)
Returns true if they look alike
'''
def checkBackPropagation(n, W, b, X, Y, epsilon = 1e-5):
  (dW1, db1) = calcGradients(W, b, X, Y)
  (dW2, db2, _) = backPropagation(n, W, b, X, Y)

  theta1 = reshapeInTheta(dW1, db1)
  theta2 = reshapeInTheta(dW2, db2)

  diff = np.linalg.norm(theta1 - theta2) / (np.linalg.norm(theta1) + np.linalg.norm(theta2))

  if diff > epsilon:
    print("\ndiff:")
    print(diff)

  return diff < epsilon

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
  endIndex = 0

  for i in range(len(WCopy)):
    wSize = WCopy[i].size
    startIndex = endIndex
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
    layerBias = np.random.randn(n[i], 1) * np.sqrt(2.)

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

  (aL, _) = forwardPropagation(W, b, x)
  maxIndex = aL.argmax()

  if x[maxIndex] == 0:
    y = np.zeros((9, 1))
    y[maxIndex] = 1

    return y

  aL[maxIndex] = 0

  maxIndex = aL.argmax()

  if x[maxIndex] == 0:
    y = np.zeros((9, 1))
    y[maxIndex] = 1

    return y

  aL[maxIndex] = 0

  maxIndex = aL.argmax()

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



'''
Receives layers structure list (e.g. [9, 9])
weights, bias, and file name. Saves all data in a file. 
'''
def saveModel(layers, W, b, fname = 'test.model'):
  assert isinstance(fname, str)
  assert isinstance(layers, list)
  assert isinstance(W, np.ndarray)
  assert isinstance(b, np.ndarray)

  flatLayers = [len(layers)]
  flatLayers.extend(layers)
  flatLayers = np.array(flatLayers)
  flatLayers = flatLayers.reshape(flatLayers.size, 1)

  for i in range(len(layers) - 1):
    flatWi = W[i].reshape(W[i].size, 1)

    flatW = flatWi if i == 0 else np.concatenate((flatW, flatWi), axis = 0)
    flatb = b[i] if i == 0 else np.concatenate((flatb, b[i]), axis = 0)

  result = np.concatenate((flatLayers, flatW, flatb))
  # print(result)

  np.savetxt(fname, result, delimiter=',')



'''
Receives file name.
Reads file content and returns dictionary
{
  'n': <layers>,
  'W': <weights>,
  'b': <bias>,
}
'''
def loadModel(fname):
  raw = np.loadtxt(fname, delimiter=',')
  raw = raw.reshape(raw.size, 1)

  L = (raw[0][0]).astype(int)
  n = raw[1:L + 1].astype(int)
  n = n.reshape(1, L)
  n = n[0]

  start = L + 1
  W = []
  b = []

  for i in range(1, L):
    size = n[i] * n[i - 1]
    end = start + size
    layerWeights = raw[start:end]
    layerWeights = layerWeights.reshape(n[i], n[i - 1])
    start = end
    W.append(layerWeights)

  for i in range(1, L):
    size = n[i]
    end = start + size
    layerBias = raw[start:end]
    layerBias = layerBias.reshape(n[i], 1)
    start = end

    b.append(layerBias)
  
  W = np.array(W)
  b = np.array(b)

  return {
    'n': n,
    'W': W,
    'b': b,
  }