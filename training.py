import numpy as np
import position


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
def makeTrainingExampleRec(initialPosition):
  initialPositionVector = position.reshapePositionInVector(initialPosition)
  # inversePosition = position.inversePosition(initialPosition)
  # inversePositionVector = position.reshapePositionInVector(inversePosition)
  randomMovement = position.makeRandomMovement(initialPosition)
  randomMovementCoords = randomMovement['coords']
  finalPosition = randomMovement['resultPosition']
  X = []
  Y = []

  if not position.isFinalPosition(finalPosition):
    opponentMovement = makeTrainingExampleRec(finalPosition)
    finalPosition = opponentMovement['finalPosition']
    X = opponentMovement['X']
    Y = opponentMovement['Y']

  X.append(initialPositionVector)

  if position.isWinPosition(finalPosition):
    Y.append(movementMatrixInVector(randomMovementCoords, 'win'))
  elif position.isLossPosition(finalPosition):
    Y.append(movementMatrixInVector(randomMovementCoords, 'loss'))
  else:
    Y.append(movementMatrixInVector(randomMovementCoords, 'draw'))
  
  return {
    'X': X,
    'Y': Y,
    'finalPosition': finalPosition,
  }


'''
Receives movement coords: [rowIndex, colIndex]
and game result (win|loss|draw)
Makes result matrix with all zeros except one value
which is 0.1, 0.5, or 1 depending on game result.
Returns this matrix reshaped in a vector 9 x 1.
'''
def movementMatrixInVector(coords, result):
  movementMatrix = np.zeros((3, 3))
  [i, j] = coords

  if result == 'win':
    movementMatrix[i][j] = 1
  elif result == 'loss':
    movementMatrix[i][j] = 0.1
  else:
    movementMatrix[i][j] = 0.5

  return position.reshapePositionInVector(movementMatrix)
