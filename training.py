import numpy as np
import position
import model



'''
Generates and saves m traning examples
'''
def generate_and_save_m_training_examples(m, make_movement_fn=None, model_fname=None, traing_fname=None):
  if make_movement_fn == None and model_fname != None:
    modelInstance = model.load(model_fname)
    W = modelInstance['W']
    b = modelInstance['b']
    make_movement_fn = lambda x: model.predict2(W, b, x)

  trainingExamples = make_training_examples(make_movement_fn)

  X = trainingExamples['X']
  Y = trainingExamples['Y']

  for i in range(0, m):
    trainingExamples = make_training_examples(make_movement_fn)
    X = np.append(X, trainingExamples['X'], axis = 1)
    Y = np.append(Y, trainingExamples['Y'], axis = 1)

    if len(X[0]) >= m:
      break

    if i % 100 == 0:
      print("Done: " + str(len(X[0])))

  X = X[:,0:m]
  Y = Y[:,0:m]

  if traing_fname == None:
    save_training_examples({
      'X': X,
      'Y': Y,
    })
  else:  
    save_training_examples({
      'X': X,
      'Y': Y,
    }, file_name=traing_fname)



'''
Receives training data x, or y, or both
Returns true if all passed training data are properly formatted.
'''
def is_proper_training_data(x = None, y = None):
  x_not_provided = not isinstance(x, np.ndarray)
  y_not_provided = not isinstance(y, np.ndarray)

  if x_not_provided and y_not_provided:
    return True

  if y_not_provided:
    return is_proper_training_X_data(x)
  
  if x_not_provided:
    return is_proper_training_Y_data(y)
  
  return is_proper_training_X_data(x) and is_proper_training_Y_data(y)



'''
Receives a vector 9x1
Returns true if this vector represents proper vector x of a training example.
This vector converted into position and inverted supposed to be a real position
'''
def is_proper_training_X_data(x):
  assert isinstance(x, np.ndarray)
  assert x.shape == (9, 1)

  pos_from_vec = position.transform_vector_into_position(x)
  inv_pos = position.invert_position(pos_from_vec)

  return position.is_real_position(inv_pos)



'''
Receives a vector 9x1
Returns true if this vector represents proper vector y of a training example.
'''
def is_proper_training_Y_data(y):
  assert isinstance(y, np.ndarray)
  assert y.shape == (9, 1)

  return (y >= 0).all() and (y <= 1).all()



'''
Receives a real game position and a function to predict movement.
Predicts the movement coords with help of make_movement_fn
(places either 1 for main player or -1 for the opponent)
Returns result position (which is either a real or a final position) and coords of the movement
'''
def make_movement(game_position, make_movement_fn, use_random_movement = False):
  assert isinstance(game_position, np.ndarray)
  assert game_position.shape == (3, 3)
  assert position.is_real_position(game_position)

  if game_position.sum() == 0:
    # main player makes movement
    position_vector = position.transform_position_into_vector(game_position)
  else:
    # opponent makes movement
    inverted_position = position.invert_position(game_position)
    position_vector = position.transform_position_into_vector(inverted_position)

  (movement_vector, highest_al) = make_movement_fn(position_vector)

  if use_random_movement:
    random_movement = position.make_random_movement(game_position)
    random_movement['highest_al'] = highest_al
    return random_movement

  movement_position = position.transform_vector_into_position(movement_vector)

  for i in range(3):
    for j in range(3):
      if (movement_position[i][j] == 1):
        coords = (i, j)
        break

  i2 = coords[0]
  j2 = coords[1]

  # choose random empty cell if make_movement_fn failed to choose an empty cell
  if game_position[i2][j2] != 0:
    random_movement = position.make_random_movement(game_position)
    random_movement['highest_al'] = highest_al
    return random_movement

  result_position = position.clone(game_position)

  result_position[i2][j2] = 1 if game_position.sum() == 0 else -1

  movement = {
    'coords': coords,
    'result_position': result_position,
    'highest_al': highest_al,
  }

  return movement



'''
Returns a few training examples:
  X: matrix 9 x m
  Y: matrix 9 x m
'''
def make_training_examples(make_movement_fn):
  zero_position = position.make_zero_position()
  main_player_starts = np.random.rand() > 0.5

  if main_player_starts:
    training_examples = make_training_examples_rec(zero_position, make_movement_fn)
  else:
    opponent_movement = make_movement(zero_position, make_movement_fn)
    non_zero_position = opponent_movement['result_position']
    training_examples = make_training_examples_rec(non_zero_position, make_movement_fn)

  return {
    'X': training_examples['X'],
    'Y': training_examples['Y'],
  }



'''
Receives initial position (real position).
Makes one or two movements: first for the main player,
and (if it is not final position) the second for the opponent.
If the result position is not the final position, then calls itself recursively.
Returns training examples based on main player movements, a dictionary:
  X: matrix 9 x m
  Y: matrix 9 x m; each column is a movement matrix reshaped in vector
    with 0.001 for taken cells
    1.0 for a movement leading to a win
    0.66 for a movement leading to a draw
    0.33 for a movement leading to a loss
'''
def make_training_examples_rec(initial_position, make_movement_fn):
  assert isinstance(initial_position, np.ndarray)
  assert initial_position.shape == (3, 3)
  assert position.is_real_position(initial_position)

  debug = False

  chance_of_random_main_player_movement = np.random.rand() < 0.05

  # main player movement could be random
  main_player_movement = make_movement(initial_position, make_movement_fn, use_random_movement=chance_of_random_main_player_movement)

  position_after = main_player_movement['result_position']

  if position.is_final_position(position_after):
    (x, y) = make_single_training_example_for_main_player(initial_position, main_player_movement, position_after)

    return {
      'X': x,
      'Y': y,
      'final_position': position_after,
    }

  # opponent movement could be random
  chance_of_random_opponent_movement = np.random.rand() < 0.01

  opponent_movement = make_movement(position_after, make_movement_fn, use_random_movement=chance_of_random_opponent_movement)

  position_after = opponent_movement['result_position']

  if position.is_final_position(position_after):
    (x, y) = make_single_training_example_for_main_player(initial_position, main_player_movement, position_after)

    return {
      'X': x,
      'Y': y,
      'final_position': position_after,
    }

  if debug:
    print("initial_position")
    position.print_position(initial_position)
    print("main_player_movement")
    print(main_player_movement)
    print("opponent_movement")
    print(opponent_movement)
    print("opponent_movement_result_position")
    position.print_position(position_after)

  the_dict = make_training_examples_rec(position_after, make_movement_fn)
  X = the_dict['X']
  Y = the_dict['Y']
  final_position = the_dict['final_position']

  assert position.is_final_position(final_position)

  (x, y) = make_single_training_example_for_main_player(initial_position, main_player_movement, final_position)

  X = np.append(X, x, axis = 1)
  Y = np.append(Y, y, axis = 1)

  if debug:
    print(X)
    print(Y)


  return {
    'X': X,
    'Y': Y,
    'final_position': final_position,
  }



'''
Receives the following:
- position before movement (a real position),
- applied movement, 
- the game result position (a final position).
The movement could be the main player (who put an X)
or for the opponent (who put an O)
Returns a single training example for the movement (x and y)
'''
def make_single_training_example_for_main_player(position_before, movement, final_position):
  assert position.is_real_position(position_before)
  assert isinstance(movement['coords'], tuple)
  assert len(movement['coords']) == 2
  assert position.is_final_position(final_position)

  debug = False

  (i, j) = movement['coords']
  # Main player is not necessarily the one who starts the game!
  main_player_started_the_game = final_position[i][j] == 1

  if main_player_started_the_game:
    x = position.transform_position_into_vector(position_before)
  else:
    position_before_inverted = position.invert_position(position_before)
    x = position.transform_position_into_vector(position_before_inverted)
  
  if debug:
    print('position_before')
    position.print_position(position_before)
    print('x')
    print(x)

  result_position = movement['result_position']
  zeros_in_result_position = (result_position == 0).astype(np.int8).sum()
  zeros_in_final_position = (final_position == 0).astype(np.int8).sum()
  is_last_game_movement = zeros_in_result_position == zeros_in_final_position
  is_prelast_game_movement = zeros_in_result_position - zeros_in_final_position == 1

  if is_last_game_movement:
    if position.is_win_position(final_position):
      # player X plays and wins
      value = 1
    elif position.is_loss_position(final_position):
      # player O plays and wins
      value = 1
    elif position.is_draw_position(final_position):
      # player X plays and draws
      value = 0.5
    else:
      # must never happen
      assert False
  elif is_prelast_game_movement:
    if position.is_win_position(final_position):
      # player X plays then O plays and wins
      value = 0.1
    elif position.is_loss_position(final_position):
      # player O plays then X plays and wins
      value = 0.1
    elif position.is_draw_position(final_position):
      # player O plays then X plays and draws
      value = 0.5
    else:
      # must never happen
      assert False
  else:
    value = movement['highest_al']

  y_as_position = (position_before != 0).astype(np.int8) * 0.001
  y_as_position[i][j] = value
  y = position.transform_position_into_vector(y_as_position)

  if debug:
    print('y')
    print(y)
    raw_input("...")

  assert is_proper_training_data(x, y)

  return (x, y)



'''
Receives initial position, movement coords: [rowIndex, colIndex]
and game result (win|loss|draw)
Makes result matrix where we put
  0.001 for taken places
  0...0.5 for the movement coords if loss
  0.5...1 for the movement coords if win
  0.5...0.75 for the movement coords if draw
  0 for all the other not taken places
Returns a vector 9 x 1.
'''
def movement_matrix_in_vector(initial_position, movement_coords, result, final_position):
  assert isinstance(initial_position, np.ndarray)
  assert initial_position.shape == (3, 3)

  assert isinstance(movement_coords, tuple)
  assert len(movement_coords) == 2

  assert isinstance(result, str)

  assert isinstance(final_position, np.ndarray)
  assert final_position.shape == (3, 3)

  debug = False

  movementMatrix = np.zeros((3, 3))
  [mi, mj] = movement_coords

  zeros_in_final_position = (final_position == 0).astype(np.int8).sum()
  zeros_in_initial_position = (initial_position == 0).astype(np.int8).sum()
  power = zeros_in_initial_position - zeros_in_final_position - 1
  reward = 0.5 / (2 ** power)

  for i in range(3):
    for j in range(3):
      if i == mi and j == mj:
        if result == 'win':
          movementMatrix[i][j] = 0.75 + 0.5 * reward
        elif result == 'loss':
          movementMatrix[i][j] = 0.5 - reward + 0.001
        else:
          movementMatrix[i][j] = 0.5 + 0.5 * reward
      elif initial_position[i][j] == 0:
        movementMatrix[i][j] = 0
      else:
        movementMatrix[i][j] = 0.001

  if debug:
    print("\nresult:")
    print(result)
    print("initial position:")
    position.print_position(initial_position)
    print("final position:")
    position.print_position(final_position)
    print("power:")
    print(power)
    print("reward:")
    print(reward)
    print("movementMatrix:")
    print(movementMatrix)
    raw_input("Enter")

  y = position.transform_position_into_vector(movementMatrix)

  # print('initial position')
  # position.printPosition(initialPosition)

  # print('y')
  # position.printMovement(y.reshape(9, 1))

  # print('final position')
  # position.printPosition(finalPosition)

  # raw_input("Press Enter to continue...")

  return y



'''
Reads and returns training examples
'''
def read_training_examples(m=0, fname='m_training_examples.csv'):
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



'''
Receives an array of m training examples.
Each training example is a dict where
  X is an array of 9 integers
  Y is an array of 9 floats
Reshapes each tr. example in 18 dimensional vector
transforms result matrix of 18 x m and saves it in file m_training_examples.csv
'''
def save_training_examples(trainingExamples, file_name='m_training_examples.csv'):
  print("file_name")
  print(file_name)
  X = trainingExamples['X']
  Y = trainingExamples['Y']
  XY = np.append(X, Y, axis = 0)
  np.savetxt(file_name, XY.T, fmt='%0.3f', delimiter=',')
