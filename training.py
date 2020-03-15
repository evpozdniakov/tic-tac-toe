import numpy as np
import position


'''
Returns a few training examples:
  X: matrix 9 x m
  Y: matrix 9 x m
'''
def makeTrainingExamples():
  zeroPosition = np.zeros((3, 3))
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

  if position.isWinPosition(finalPosition):
    if randomMovementPutX:
      x = initialPositionVector
      y = movementMatrixInVector(randomMovementCoords, 'win')
    else:
      x = inverseInitialPositionVector
      y = movementMatrixInVector(randomMovementCoords, 'loss')
  elif position.isLossPosition(finalPosition):
    if randomMovementPutX:
      x = initialPositionVector
      y = movementMatrixInVector(randomMovementCoords, 'loss')
    else:
      x = inverseInitialPositionVector
      y = movementMatrixInVector(randomMovementCoords, 'win')
  else:
    if randomMovementPutX:
      x = initialPositionVector
    else:
      x = inverseInitialPositionVector
    y = movementMatrixInVector(randomMovementCoords, 'draw')
  
  return (x, y)



'''
Receives movement coords: [rowIndex, colIndex]
and game result (win|loss|draw)
Makes result matrix with all zeros except one value
which is 0.1, 0.5, or 1 depending on game result.
Returns this matrix reshaped in a vector 9 x 1.
'''
def movementMatrixInVector(coords, result):
  assert isinstance(result, str)

  assert isinstance(coords, tuple)
  assert len(coords) == 2

  movementMatrix = np.zeros((3, 3))
  [i, j] = coords

  if result == 'win':
    movementMatrix[i][j] = 1
  elif result == 'loss':
    movementMatrix[i][j] = 0.1
  else:
    movementMatrix[i][j] = 0.5

  return position.reshapePositionInVector(movementMatrix)
