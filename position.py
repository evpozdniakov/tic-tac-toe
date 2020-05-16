import numpy as np
from termcolor import colored


# A position is a matrix 3x3 where
#   0 is an empty cell
#   1 is the main player movement "X"
#   -1 is the opponent player movement "O"

'''
Receives game position and a movement vector 9x1.
Clones the position and changes it by placing 1 according
to the movement vector. Returns changed position.
'''
def change_position_with_movement_vector(game_position, movement):
  assert isinstance(game_position, np.ndarray)
  assert game_position.shape == (3, 3)
  assert isinstance(movement, np.ndarray)
  assert movement.shape == (9, 1)

  cloned_position = clone(game_position)
  result_position = cloned_position + movement.reshape(3, 3)

  assert game_position.sum() == result_position.sum() - 1

  return result_position



'''
Receives a position.
Creates and returns a copy of the position.
'''
def clone(position):
  return np.multiply(np.ones((3, 3)), position).astype(np.int8)



'''
Receives position
Returns True if it has zeros
'''
def has_empty_cells(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  return np.square(position).sum() < 9



'''
Receives game position and returns it inverted
'''
def invert_position(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  return position * -1



'''
Returns game position and returns True if it is a real not finalized position
Real means position where the first movement was X (1)
so number of X (1) is either equal to number of O (0)
or one time bigger
Not finalized means that it is not win, loss or draw
'''
def is_real_position(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  if position.sum() < 0:
    return False
  
  if position.sum() > 1:
    return False

  if is_win_position(position):
    return False

  if is_loss_position(position):
    return False

  if is_draw_position(position):
    return False

  return True



'''
Receives game position and returns true if it is a win
(there are 3 X in a row)
'''
def is_win_position(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  if np.multiply([[1, 0, 0], [0, 1, 0], [0, 0, 1]], position).sum() == 3:
    return True
  elif np.multiply([[0, 0, 1], [0, 1, 0], [1, 0, 0]], position).sum() == 3:
    return True
  elif np.multiply([[1, 1, 1], [0, 0, 0], [0, 0, 0]], position).sum() == 3:
    return True
  elif np.multiply([[0, 0, 0], [1, 1, 1], [0, 0, 0]], position).sum() == 3:
    return True
  elif np.multiply([[0, 0, 0], [0, 0, 0], [1, 1, 1]], position).sum() == 3:
    return True
  elif np.multiply([[1, 0, 0], [1, 0, 0], [1, 0, 0]], position).sum() == 3:
    return True
  elif np.multiply([[0, 1, 0], [0, 1, 0], [0, 1, 0]], position).sum() == 3:
    return True
  elif np.multiply([[0, 0, 1], [0, 0, 1], [0, 0, 1]], position).sum() == 3:
    return True
  return False



'''
Receives game position and returns true if it is a loss
(there are 3 O in a row)
'''
def is_loss_position(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  if is_win_position(position):
    return False

  return is_win_position(invert_position(position))



'''
Receives game position and returns true 
if there is no empty place for a movement
'''
def is_draw_position(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  if is_win_position(position):
    return False
  
  if is_loss_position(position):
    return False

  if has_empty_cells(position):
    return False

  return True



'''
Receives game position.
Returns true if it is a win, or a loss, or a draw.
'''
def is_final_position(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  if is_win_position(position):
    return True

  if is_loss_position(position):
    return True
  
  if is_draw_position(position):
    return True

  return False



'''
Returns true if the position has no movements yet.
(All cells equal to zero.)
'''
def is_zero_position(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  return np.absolute(position).sum() == 0



'''
Receives real game position and returns coords of a random opponent's movement
(places either 1 for main player or -1 for the opponent)
Returns result position (which is either a real or a final position) and coords of the movement
'''
def make_random_movement(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)
  assert is_real_position(position)


  emptyCellCoords = []

  for i in range(0, 3):
    for j in range(0, 3):
      if position[i][j] == 0:
        emptyCellCoords.append((i, j))

  randomIndex = np.floor(np.random.rand() * len(emptyCellCoords)).astype(int)
  coords = emptyCellCoords[randomIndex]
  i2 = coords[0]
  j2 = coords[1]

  result_position = clone(position)

  result_position[i2][j2] = 1 if position.sum() == 0 else -1

  assert is_real_position(result_position) or is_final_position(result_position)

  movement = {
    'coords': coords,
    'result_position': result_position,
  }

  return movement



'''
Creates and returns zero position.
'''
def make_zero_position():
  return np.zeros((3, 3)).astype(np.int8)



'''
Receives a position
and prints it out
'''
def print_position(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  xo = [['_', '_', '_'],
    ['_', '_', '_'],
    ['_', '_', '_']]

  for i in range(0, 3):
    for j in range(0, 3):
      if position[i][j] == 1:
        xo[i][j] = colored('X', 'green')
      elif position[i][j] == -1:
        xo[i][j] = colored('O', 'blue')
    print(xo[i][0] + ' ' + xo[i][1] + ' ' + xo[i][2])



'''
Receives movement vector
and prints it out
'''
def print_movement(movement):
  assert isinstance(movement, np.ndarray)
  assert movement.shape == (9, 1)

  xo = [['_', '_', '_'],
    ['_', '_', '_'],
    ['_', '_', '_']]

  movement2 = (movement * 1000).astype(np.int)

  for i in range(0, 3):
    for j in range(0, 3):
      value = movement2[i * 3 + j][0]
      if value > 0:
        strValue = str(value)
        xo[i][j] = colored(strValue, 'red')
    print(xo[i][0] + ' ' + xo[i][1] + ' ' + xo[i][2])



'''
Returns real random position
'''
def randomPosition():
  m1 = np.random.rand(3, 3)
  m2 = m1 * 3
  m3 = np.floor(m2).astype(np.int8)
  m4 = m3 - 1

  if is_real_position(m4):
    return m4

  return randomPosition()



'''
Receives game position, matrix 3 x 3
Reshapes it in a vector 9 x 1 and returns.
'''
def transform_position_into_vector(position):
  assert isinstance(position, np.ndarray)
  assert position.shape == (3, 3)

  return position.reshape(9, 1)



'''
Receives vector
Returns position
'''
def transform_vector_into_position(vector):
  assert isinstance(vector, np.ndarray)
  assert vector.size == 9

  return vector.reshape(3, 3)
