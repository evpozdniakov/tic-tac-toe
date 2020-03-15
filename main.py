import numpy as np
import training
import model
import position


'''
  startPosition = np.zeros((3, 3))

  trainingExamples = training.makeTrainingExampleRec(startPosition)

  print('training examples X')
  print(trainingExamples['X'])

  print('training examples Y')
  print(trainingExamples['Y'])
'''

'''
  X = trainingExamples['X']
  Y = trainingExamples['Y']
  m = len(X)

  print('m=' + str(m))

  for i in range(0, m):
    print(str(i) + ':')
    print(X[i].reshape(3, 3))
    print(Y[i].reshape(3, 3))
    print('')
'''


n = [9, 9]
(W, b) = model.initializeWeights(n)
alpha = 0.03
# alphaDecayRate = 0.001

for i in range(0, 1000):
  trainingExamples = training.makeTrainingExamples()
  X = trainingExamples['X']
  Y = trainingExamples['Y']

  (dW, db) = model.calcGradients(W, b, X, Y)

  # alpha *= 1 / (1 + alphaDecayRate * i)
  model.updateWeights(W, dW, b, db, alpha)

  if i % 100 == 0:
    print('iteration ' + str(i))
    AL = model.forwardPropagation(W, b, X)
    cost = model.costFunction(Y, AL)
    print('cost')
    print(cost)
    print('alpha')
    print(alpha)
