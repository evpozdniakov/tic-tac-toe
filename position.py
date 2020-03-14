import numpy as np


'''
Receives a position
and prints it out
'''
def printPosition(position):
  assert position.shape == (3, 3)
  print(position)



'''
Returns random position
'''
def randomPosition():
  m1 = np.random.rand(3, 3)
  m2 = m1 * 3
  m3 = np.floor(m2)
  m4 = m3 - 1

  if isRealPosition(m4):
    return m4

  return randomPosition()


'''
Receives game position and returns it inverted
'''
def inversePosition(position):
  return position * -1


'''
Returns game position and returns True if it is a real position
'''
def isRealPosition(position):
  if position.sum() == 0:
    return True
  
  if position.sum() == 1:
    return True
  
  return False


'''
Receives game position and returns true if it is a win
(there are 3 X in a row)
'''
def isWinPosition(position):
  if np.multiply([[1, 0, 0], [0, 1, 0], [0, 0, 1]], position).sum() == 3:
    print('1')
    return True
  elif np.multiply([[0, 0, 1], [0, 1, 0], [1, 0, 0]], position).sum() == 3:
    print('2')
    return True
  elif np.multiply([[1, 1, 1], [0, 0, 0], [0, 0, 0]], position).sum() == 3:
    print('3')
    return True
  elif np.multiply([[0, 0, 0], [1, 1, 1], [0, 0, 0]], position).sum() == 3:
    print('4')
    return True
  elif np.multiply([[0, 0, 0], [0, 0, 0], [1, 1, 1]], position).sum() == 3:
    print('5')
    return True
  elif np.multiply([[1, 0, 0], [1, 0, 0], [1, 0, 0]], position).sum() == 3:
    print('6')
    return True
  elif np.multiply([[0, 1, 0], [0, 1, 0], [0, 1, 0]], position).sum() == 3:
    print('7')
    return True
  elif np.multiply([[0, 0, 1], [0, 0, 1], [0, 0, 1]], position).sum() == 3:
    print('8')
    return True
  return False


'''
Receives game position and returns true if it is a loss
(there are 3 O in a row)
'''
def isLossPosition(position):
  if isWinPosition(position):
    return False

  return isWinPosition(inversePosition(position))


'''
Receives position
Returns True if it has zeros
'''
def positionHasEmptyCells(position):
  return np.square(position).sum() < 9


'''
Receives game position and returns true 
if there is no empty place for a movement
'''
def isDrawPosition(position):
  if isWinPosition(position):
    return False
  
  if isLossPosition(position):
    return False

  if positionHasEmptyCells(position):
    return False

  return False


'''
Receives game position.
Returns true if it is a win, or a loss, or a draw.
'''
def isFinalPosition(position):
  if isWinPosition(position):
    return True

  if isLossPosition(position):
    return True
  
  if isDrawPosition(position):
    return True

  return False


'''
Receives game position and returns random movement.
A position is a matrix 3x3 where
  0 is an empty cell
  1 is main player X
  -1 is opponent player O
'''
def makeRandomMovement(position):
  assert not isDrawPosition(position)

  emptyCellCoords = []

  for i in range(0, 3):
    for j in range(0, 3):
      if position[i][j] == 0:
        emptyCellCoords.append((i, j))

  randomIndex = np.floor(np.random.rand() * len(emptyCellCoords)).astype(int)
  print('======= randomIndex:')
  print(randomIndex)
  coords = emptyCellCoords[randomIndex]
  i2 = coords[0]
  j2 = coords[1]

  # TODO make copyPosition
  resultPosition = np.multiply(np.ones((3, 3)), position)

  resultPosition[i2][j2] = 1 if position.sum() == 0 else -1

  movement = {
    'coords': coords,
    'resultPosition': resultPosition,
  }

  return movement



'''
Receives game position, matrix 3 x 3
Reshapes it in a vector 9 x 1 and returns.
'''
def reshapePositionInVector(position):
  return position.reshape(9, 1)