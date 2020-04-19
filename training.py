import numpy as np
import position


'''
Returns a few training examples:
  X: matrix 9 x m
  Y: matrix 9 x m
'''
def makeTrainingExamples():
  zeroPosition = np.zeros((3, 3)).astype(np.int8)
  trainingExampes = makeTrainingExamplesRec(zeroPosition)

  return {
    'X': trainingExampes['X'],
    'Y': trainingExampes['Y'],
  }



'''
Receives initial position, makes random movement.
If the movement does not give final position (win, loss, or draw)
then calls itself recursively.

Returns a dictionary with
  X: matrix 9 x m
  Y: matrix 9 x m; each column is a movement matrix reshaped in vector
    with all zeros except movement coords, which is
    0.05 if movement resulted a loss,
    0.5 if movement resulted a draw,
    1 if it resulted a win.
  finalPosition: final game position
'''
def makeTrainingExamplesRec(initialPosition):
  assert isinstance(initialPosition, np.ndarray)
  assert initialPosition.shape == (3, 3)

  randomMovement = position.makeRandomMovement(initialPosition)
  finalPosition = randomMovement['resultPosition']

  if position.isFinalPosition(finalPosition):
    (x, y) = makeSingleTrainingExample(initialPosition, randomMovement, finalPosition)

    return {
      'X': x,
      'Y': y,
      'finalPosition': finalPosition,
    }

  opponentMovement = makeTrainingExamplesRec(finalPosition)
  finalPosition = opponentMovement['finalPosition']
  (x, y) = makeSingleTrainingExample(initialPosition, randomMovement, finalPosition)

  X = np.append(opponentMovement['X'], x, axis = 1)
  Y = np.append(opponentMovement['Y'], y, axis = 1)

  return {
    'X': X,
    'Y': Y,
    'finalPosition': finalPosition,
  }



'''
Receives initial position and random movement.
Also receives the final position (game result).
Returns single training example (x and y)
'''
def makeSingleTrainingExample(initialPosition, randomMovement, finalPosition):
  assert isinstance(initialPosition, np.ndarray)
  assert initialPosition.shape == (3, 3)

  assert isinstance(randomMovement['coords'], tuple)
  assert len(randomMovement['coords']) == 2

  assert isinstance(finalPosition, np.ndarray)
  assert finalPosition.shape == (3, 3)

  initialPositionVector = position.reshapePositionInVector(initialPosition)
  inverseInitialPosition = position.inversePosition(initialPosition)
  inverseInitialPositionVector = position.reshapePositionInVector(inverseInitialPosition)
  randomMovementCoords = randomMovement['coords']
  (rowIndex, colIndex) = randomMovementCoords
  randomMovementPutX = finalPosition[rowIndex][colIndex] == 1
  inverseFinalPosition = position.inversePosition(finalPosition)

  if position.isWinPosition(finalPosition):
    if randomMovementPutX:
      x = initialPositionVector
      y = movementMatrixInVector(initialPosition, randomMovementCoords, 'win', finalPosition)
    else:
      x = inverseInitialPositionVector
      y = movementMatrixInVector(inverseInitialPosition, randomMovementCoords, 'loss', inverseFinalPosition)
  elif position.isLossPosition(finalPosition):
    if randomMovementPutX:
      x = initialPositionVector
      y = movementMatrixInVector(initialPosition, randomMovementCoords, 'loss', finalPosition)
    else:
      x = inverseInitialPositionVector
      y = movementMatrixInVector(inverseInitialPosition, randomMovementCoords, 'win', inverseFinalPosition)
  else:
    if randomMovementPutX:
      x = initialPositionVector
      y = movementMatrixInVector(initialPosition, randomMovementCoords, 'draw', finalPosition)
    else:
      x = inverseInitialPositionVector
      y = movementMatrixInVector(inverseInitialPosition, randomMovementCoords, 'draw', inverseFinalPosition)
  
  return (x, y)



'''
Receives initial position, movement coords: [rowIndex, colIndex]
and game result (win|loss|draw)
Makes result matrix where we put
  0 for the places taken by X/O
  0.1 for the movement coords if loss
  1 for the movement coords if win
  0.5 for the movement coords if draw
  0.3 for all the other not taken places
Returns a vector 9 x 1.
'''
def movementMatrixInVector(initialPosition, movementCoords, result, finalPosition):
  assert isinstance(initialPosition, np.ndarray)
  assert initialPosition.shape == (3, 3)

  assert isinstance(movementCoords, tuple)
  assert len(movementCoords) == 2

  assert isinstance(result, str)

  assert isinstance(finalPosition, np.ndarray)
  assert finalPosition.shape == (3, 3)

  movementMatrix = np.zeros((3, 3))
  [mi, mj] = movementCoords

  for i in range(3):
    for j in range(3):
      if i == mi and j == mj:
        if result == 'win':
          movementMatrix[i][j] = 1
        elif result == 'loss':
          movementMatrix[i][j] = 0.1
        else:
          movementMatrix[i][j] = 0.5
      elif initialPosition[i][j] == 0:
        movementMatrix[i][j] = 0.3

  y = position.reshapePositionInVector(movementMatrix)

  # print('initial position')
  # position.printPosition(initialPosition)

  # print('y')
  # position.printMovement(y.reshape(9, 1))

  # print('final position')
  # position.printPosition(finalPosition)

  # raw_input("Press Enter to continue...")

  return y



'''
Receives an array of m training examples.
Each training example is a dict where
  X is an array of 9 integers
  Y is an array of 9 floats
Reshapes each tr. example in 18 dimensional vector
transforms result matrix of 18 x m and saves it in file m_training_examples.csv
'''
def saveTrainingExamples(trainingExamples, fileName = 'm_training_examples.csv'):
  X = trainingExamples['X']
  Y = trainingExamples['Y']
  XY = np.append(X, Y, axis = 0)
  np.savetxt(fileName, XY.T, fmt='%0.1f', delimiter=',')



'''
Generates and saves m traning examples
'''
def generateAndSaveTrainingExamples(m):
  trainingExamples = makeTrainingExamples()

  X = trainingExamples['X']
  Y = trainingExamples['Y']

  for _ in range(0, m):
    trainingExamples = makeTrainingExamples()
    X = np.append(X, trainingExamples['X'], axis = 1)
    Y = np.append(Y, trainingExamples['Y'], axis = 1)
    if len(X[0]) >= m:
      break

  X = X[:,0:m]
  Y = Y[:,0:m]
  
  saveTrainingExamples({
    'X': X,
    'Y': Y,
  })



'''
Reads and returns training examples
'''
def readTrainingExamples(m=0, fname='m_training_examples.csv'):
  raw = np.loadtxt(fname, delimiter=',')
  XY = raw.T
  X = XY[0:9, :].astype(np.int8)
  Y = XY[9:18, :].astype(np.float32)

  if m > 0:
    return {
      'X': X[:, 0:m],
      'Y': Y[:, 0:m],
    }

  return {
    'X': X,
    'Y': Y,
  }
