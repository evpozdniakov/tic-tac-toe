import numpy as np
import training
import model
import position


def trainModelScenario1():
  n = [9, 18, 18, 18, 9]
  (W, b) = model.initializeWeights(n)
  alpha = 0.01
  alphaDecayRate = 0.00001

  for i in range(0, 1000):
    trainingExamples = training.makeTrainingExamples()
    X = trainingExamples['X']
    Y = trainingExamples['Y']

    (dW, db) = model.calcGradients(W, b, X, Y)

    alpha *= 1 / (1 + alphaDecayRate * i)
    model.updateWeights(W, dW, b, db, alpha)

    if i % 100 == 0:
      print('iteration ' + str(i))
      AL = model.forwardPropagation(W, b, X)
      cost = model.costFunction(Y, AL)
      print('alpha')
      print(alpha)
      print('cost')
      print(cost)
  # '''
  initPos = np.zeros((3, 3))
  initPos[1][2] = -1

  for i in range(0, 10):
    print('======== initPos')
    position.printPosition(initPos)
    # print('initPos')
    # print(initPos)

    initPosVector = position.reshapePositionInVector(initPos)
    # print('initPosVector')
    # print(initPosVector)

    movementVector = model.predict(W, b, initPosVector)
    # print('nextPosVector')
    # print(movementVector)

    nextPos = (initPosVector + movementVector).reshape(3, 3)
    position.printPosition(nextPos)
    print('nextPos ========')
    print('')

    if position.isFinalPosition(nextPos):
      break

    initPos = position.inversePosition(nextPos)

  print('------ end -------')
  # '''



def trainModelScenario2(n, fname, alpha = 0.001, iterations = 10000):
  (W, b) = model.initializeWeights(n)
  ex = training.readTrainingExamples(100)

  X = ex['X']
  assert X.shape == (9, 100)

  Y = ex['Y']
  assert Y.shape == (9, 100)


  # L is a number of NN layers
  # (L = 3 for a model 9x18x18x9)
  L = len(W)
  assert len(W) == L


  for i in range(0, iterations):
    # (dW, db) = model.calcGradients(W, b, X, Y)
    (dW, db, A) = model.calcGradients2(n, W, b, X, Y)

    model.updateWeights(W, dW, b, db, alpha)

    if i % 10 == 0:
      print('iteration ' + str(i))
      # AL = model.forwardPropagation(W, b, X)
      # cost = model.costFunction(Y, AL)
      cost = model.costFunction2(Y, A[L])
      # print('alpha')
      # print(alpha)
      print('cost')
      print(cost)

  print('------ end -------')

  model.saveModel(n, W, b, fname)
  # res = model.loadModel('9x9.model')
  # print(res['n'])
  # print(W.shape)
  # print(W)
  # print(res['W'].shape)
  # print(res['W'])



def testModel(fname):
  model9x9 = model.loadModel(fname)

  # n = model9x9['n']
  W = model9x9['W']
  b = model9x9['b']

  for _ in range(100):
    randomPosition = position.randomPosition()

    position.printPosition(randomPosition)

    x = position.reshapePositionInVector(randomPosition)

    print(' ')

    movement = model.predict(W, b, x)

    position.printMovement(movement)

    raw_input("Press Enter to continue...")


trainModelScenario2(n = [9, 81, 81, 9], fname = '9x81x81x9.model', alpha = 0.01, iterations=2000)


# testModel('9x9x9x9.model')
# testModel('9x18x18x9.model')
# testModel('9x81x81x9.model')


# training.generateAndSaveTrainingExamples(1000)
'''
ex = training.readTrainingExamples()
X = ex['X']
Y = ex['Y']

for _ in range(100):
  randomIndex = np.floor(np.random.rand() * 1000)
  x = X[0:9, randomIndex]
  print('x')
  print(x)
  position.printPosition(x.reshape(3, 3))
  y = Y[0:9, randomIndex]
  print('y')
  position.printMovement(y.reshape(9, 1))
  raw_input("Press Enter to continue...")
'''

