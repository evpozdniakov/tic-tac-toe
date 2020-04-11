import numpy as np



'''
Receives NN weights, bias, and initial entries matrix 9 x m
Calculates and returns the result 9 x m vector
'''
def forwardPropagation(W, b, X):
  A = forwardPropagation2(W, b, X)

  L = len(W)

  return A[L]



'''
Receives NN weights, bias, and initial entries matrix 9 x m
Calculates and returns A for each layer
'''
def forwardPropagation2(W, b, X):
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

  return A



'''
Receives Y and Yhat
(both are numpy arrays 9 x m)
Calculates the cost based on the following algorithm:
- only Yhat with value 1 (the movement) is considered
'''
def costFunction(Y, Yhat):
  assert isinstance(Y, np.ndarray)
  assert len(Y.shape) == 2
  assert isinstance(Yhat, np.ndarray)
  assert Y.shape[0] == 9
  assert Y.shape == Yhat.shape

  m = Y.shape[1]

  # Yhat is a matrix 9xm
  # YhatVmax is a matrix 1xm with max value in each column of Yhat
  YhatVmax = np.max(Yhat, axis = 0)
  # Mast is a matrix 9xm with zeros ans a single 1 in each column
  Mask = (Yhat == YhatVmax).astype(int)
  assert np.sum(Mask) == m


  YMask = Y * Mask
  YhatMask = Yhat * Mask

  YhatMaskFlat = np.sum(YhatMask, axis = 0).reshape(1, m)
  YMaskFlat = np.sum(YMask, axis = 0).reshape(1, m)

  cost = YMaskFlat * np.log(YhatMaskFlat) + (1 - YMaskFlat) * np.log(1 - YhatMaskFlat)

  cost = np.sum(cost)
  cost = cost * -1 / m

  return cost



'''
Receives Y and Yhat
(both are numpy arrays 9 x m)
Calculates classic cost function and returns it
'''
def costFunction2(Y, Yhat):
  assert isinstance(Y, np.ndarray)
  assert len(Y.shape) == 2
  assert isinstance(Yhat, np.ndarray)
  assert Y.shape[0] == 9
  assert Y.shape == Yhat.shape

  m = Y.shape[1]

  cosmMatrix = (Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))

  cost = np.sum(cosmMatrix) * -1.0 / m

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
Receives layers, weights, bias, initial entries, and Y
Calculates and returns derivatives for each weight and bias
'''
def calcGradients2(n, W, b, X, Y):
  L = len(W)
  assert len(W) == L

  m = Y.shape[1]

  # calculate A for all layers
  A = forwardPropagation2(W, b, X)

  # create dW and db
  dW = np.array([np.zeros(W[0].shape), np.zeros(W[1].shape), np.zeros(W[2].shape)])
  db = np.array([np.zeros(b[0].shape), np.zeros(b[1].shape), np.zeros(b[2].shape)])

  for i in range(m):
    a3i = A[3].T[i].reshape(1, len(A[3]))
    # assert a3i.shape == (1, 9)

    yi = Y.T[i].reshape(1, len(Y))
    # assert yi.shape == (1, 9)

    dz3i = a3i - yi
    # assert dz3i.shape == (1, 9)

    # assert dW[2].shape == (9, 18)
    # assert len(dW[2]) == 9

    a2i = A[2].T[i].reshape(1, len(dW[2][0])) # 1, 18
    # assert a2i.shape == (1, 18)

    dw2i = np.dot(dz3i.T, a2i)
    # assert dw2i.shape == (9, 18)
    # assert dW[2].shape == (9, 18)

    dW[2] += dw2i
    # assert dW[2].shape == (9, 18)

    db[2] += dz3i.T
    # assert db[2].shape == (9, 1)

    dz2i = a2i * (1 - a2i)
    # assert dz2i.shape == (1, 18)

    a1i = A[1].T[i].reshape(1, len(dW[1][0])) # 1, 18
    # assert a1i.shape == (1, 18)

    dw1i = np.dot(dz2i.T, a1i)
    # assert dw1i.shape == (18, 18)

    dW[1] += dw1i
    # assert dW[1].shape == (18, 18)

    db[1] += dz2i.T
    # assert db[1].shape == (18, 1)

    dz1i = a1i * (1 - a1i)
    # assert dz1i.shape == (1, 18)

    a0i = A[0].T[i].reshape(1, len(dW[0][0])) # 1, 9
    # assert a0i.shape == (1, 9)

    dw0i = np.dot(dz1i.T, a0i)
    # assert dw0i.shape == (18, 9)

    dW[0] += dw0i
    # assert dW[0].shape == (18, 9)

    db[0] += dz1i.T
    # assert db[0].shape == (18, 1)

  dW = dW / m
  db = db / m

  return (dW, db, A)



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