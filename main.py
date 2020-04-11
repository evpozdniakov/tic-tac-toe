import numpy as np
import training
import model
import position


def trainModelScenario(n, fname, alpha = 0.001, iterations = 10000):
  (W, b) = model.initializeWeights(n)
  ex = training.readTrainingExamples(24)

  X = ex['X']
  # assert X.shape == (9, 500)

  Y = ex['Y']
  # assert Y.shape == (9, 500)


  # L is a number of NN layers
  # (L = 3 for a model 9x18x18x9)
  L = len(W)
  assert len(W) == L


  for i in range(0, iterations):
    # (dW, db) = model.calcGradients(W, b, X, Y)
    (dW, db, A) = model.backPropagation(n, W, b, X, Y)

    model.updateWeights(W, dW, b, db, alpha)

    if i % 10 == 0:
      print('iteration ' + str(i))
      AL = model.forwardPropagation(W, b, X)
      # cost = model.costFunction(Y, AL)
      cost = model.costFunction(Y, A[L])
      # print('alpha')
      # print(alpha)
      print('cost')
      print(cost)

    if i % 2 == 0:
      is_back_prop_correct = model.checkBackPropagation(n, W, b, X, Y)

      if not is_back_prop_correct:
        print("BP is not correct")
        exit()

  print('------ end -------')

  model.saveModel(n, W, b, fname)
  # res = model.loadModel('9x9.model')
  # print(res['n'])
  # print(W.shape)
  # print(W)
  # print(res['W'].shape)
  # print(res['W'])



def testModel(fname):
  modelInstance = model.loadModel(fname)

  # n = model9x9['n']
  W = modelInstance['W']
  b = modelInstance['b']

  trainingExamples = training.readTrainingExamples(100)

  for i in range(100):
    x = trainingExamples['X'].T[i]
    nextPosition = position.positionFromVector(x)

    position.printPosition(nextPosition)

    print(' predicted ')

    x = position.reshapePositionInVector(nextPosition)
    movement = model.predict(W, b, x)
    position.printMovement(movement)

    print(' expected ')

    y = trainingExamples['Y'].T[i]
    position.printMovement(y.reshape(9, 1))

    raw_input("Press Enter to continue...")


trainModelScenario(n = [9, 9, 9, 9], fname = '9x9x9x9.model', alpha = 0.01, iterations=2000)


# testModel('9x9.model')
testModel('9x9x9x9.model')
# testModel('9x81x81x9.model')
# training.generateAndSaveTrainingExamples(1000)
