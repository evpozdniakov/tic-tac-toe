import numpy as np
import position
import model


'''
Receives training data x, or y, or both
Returns true if all passed training data are properly formatted.
'''
def isProperTrainingData(x = None, y = None):
  if x == None and y == None:
    return True

  if x != None and y == None:
    return isProperTrainingXData(x)
  
  if x == None and y != None:
    return isProperTrainingYData(y)
  
  return isProperTrainingXData(x) and isProperTrainingYData(y)

'''
Receives a vector 9x1
Returns true if this vector represents proper vector x of a training example.
This vector converted into position and inverted supposed to be a real position
'''
def isProperTrainingXData(x):
  assert isinstance(x, np.ndarray)
  assert x.shape == (9, 1)

  pos_from_vec = position.transform_vector_into_position(x)
  inv_pos = position.invert_position(pos_from_vec)

  return position.is_real_position(inv_pos)



'''
Receives a vector 9x1
Returns true if this vector represents proper vector y of a training example.
'''
def isProperTrainingYData(y):
  assert isinstance(y, np.ndarray)
  assert y.shape == (9, 1)

  return (y >= 0).all() and (y <= 1).all()



'''
Receives a real game position and a function to predict movement.
Predicts the movement coords with help of make_movement_fn
(places either 1 for main player or -1 for the opponent)
Returns result position (which is either a real or a final position) and coords of the movement
'''
def makeMovement(game_position, make_movement_fn):
  assert isinstance(game_position, np.ndarray)
  assert game_position.shape == (3, 3)
  assert position.is_real_position(game_position)

  use_random_movement = np.random.rand() < 0.05
  # use_random_movement = True

  if make_movement_fn == None or use_random_movement:
    return position.make_random_movement(game_position)

  if game_position.sum() == 0:
    # main player makes movement
    position_vector = position.transform_position_into_vector(game_position)
  else:
    # opponent makes movement
    inverted_position = position.invert_position(game_position)
    position_vector = position.transform_position_into_vector(inverted_position)

  movement_vector = make_movement_fn(position_vector)
  movement_position = position.transform_vector_into_position(movement_vector)

  if (movement_position.sum() != 1):
    return position.make_random_movement(game_position)

  for i in range(3):
    for j in range(3):
      if (movement_position[i][j] == 1):
        coords = (i, j)
        break

  i2 = coords[0]
  j2 = coords[1]

  # choose random empty cell if make_movement_fn not specified
  # or if it failed to choose an empty cell
  if game_position[i2][j2] != 0:
    return position.make_random_movement(game_position)

  resultPosition = position.clone(game_position)

  resultPosition[i2][j2] = 1 if game_position.sum() == 0 else -1

  movement = {
    'coords': coords,
    'resultPosition': resultPosition,
  }

  return movement



'''
Returns a few training examples:
  X: matrix 9 x m
  Y: matrix 9 x m
'''
def makeTrainingExamples(make_movement_fn=None):
  start_position = np.zeros((3, 3)).astype(np.int8)
  use_zero_position = np.random.rand() > 0.5
  # use_zero_position = True

  if use_zero_position:
    # print("WE ARE USING ZERO POZITION")
    trainingExampes = makeTrainingExamplesRec(start_position, make_movement_fn)
  else:
    # print("WE ARE USING NON-ZERO POZITION")
    emptyCellCoords = []

    for i in range(3):
      for j in range(3):
        emptyCellCoords.append((i, j))

    randomIndex = np.floor(np.random.rand() * 9).astype(int)
    coords = emptyCellCoords[randomIndex]

    i2 = coords[0]
    j2 = coords[1]

    start_position[i2][j2] = 1

    trainingExampes = makeTrainingExamplesRec(start_position, make_movement_fn)

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
def makeTrainingExamplesRec(initialPosition, make_movement_fn):
  assert isinstance(initialPosition, np.ndarray)
  assert initialPosition.shape == (3, 3)
  assert position.is_real_position(initialPosition)

  debug = False

  some_movement = makeMovement(initialPosition, make_movement_fn)

  result_position = some_movement['resultPosition']

  if position.is_final_position(result_position):
    (x, y) = makeSingleTrainingExample(initialPosition, some_movement, result_position)

    return {
      'X': x,
      'Y': y,
      'finalPosition': result_position,
    }

  opponent_movement = makeTrainingExamplesRec(result_position, make_movement_fn)
  result_position = opponent_movement['finalPosition']
  (x, y) = makeSingleTrainingExample(initialPosition, some_movement, result_position)

  if debug:
    print("initialPosition")
    position.print_position(initialPosition)
    print("some_movement")
    print(some_movement)
    print("result_position")
    position.print_position(result_position)

  X = np.append(opponent_movement['X'], x, axis = 1)
  Y = np.append(opponent_movement['Y'], y, axis = 1)

  if debug:
    print(X)
    print(Y)

  assert position.is_final_position(result_position)

  return {
    'X': X,
    'Y': Y,
    'finalPosition': result_position,
  }



'''
Receives initial position (a real position), applied movement,
and the final position (game result).
Returns single training example (x and y)
'''
def makeSingleTrainingExample(initialPosition, movement, finalPosition):
  assert position.is_real_position(initialPosition)

  assert isinstance(movement['coords'], tuple)
  assert len(movement['coords']) == 2

  assert position.is_final_position(finalPosition)

  debug = False

  if debug:
    print("initialPosition")
    print(initialPosition)
    print("movement")
    print(movement)
    print("finalPosition")
    print(finalPosition)

  initialPositionVector = position.transform_position_into_vector(initialPosition)
  inverseInitialPosition = position.invert_position(initialPosition)
  inverseInitialPositionVector = position.transform_position_into_vector(inverseInitialPosition)
  randomMovementCoords = movement['coords']
  (rowIndex, colIndex) = randomMovementCoords

  if debug:
    print("randomMovementCoords")
    print(randomMovementCoords)
    print("rowIndex")
    print(rowIndex)
    print("colIndex")
    print(colIndex)

  randomMovementPutX = finalPosition[rowIndex][colIndex] == 1
  inverseFinalPosition = position.invert_position(finalPosition)

  if position.is_win_position(finalPosition):
    if randomMovementPutX:
      x = initialPositionVector
      y = movementMatrixInVector(initialPosition, randomMovementCoords, 'win', finalPosition)
    else:
      x = inverseInitialPositionVector
      y = movementMatrixInVector(inverseInitialPosition, randomMovementCoords, 'loss', inverseFinalPosition)
  elif position.is_loss_position(finalPosition):
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

  if debug:
    # if not isProperTrainingData(x, y):
    print("x")
    print(x)
    print("y")
    print(y)

    raw_input("...")

  assert isProperTrainingData(x, y)

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
def movementMatrixInVector(initialPosition, movementCoords, result, finalPosition):
  assert isinstance(initialPosition, np.ndarray)
  assert initialPosition.shape == (3, 3)

  assert isinstance(movementCoords, tuple)
  assert len(movementCoords) == 2

  assert isinstance(result, str)

  assert isinstance(finalPosition, np.ndarray)
  assert finalPosition.shape == (3, 3)

  debug = False

  movementMatrix = np.zeros((3, 3))
  [mi, mj] = movementCoords

  zeros_in_final_position = (finalPosition == 0).astype(np.int8).sum()
  zeros_in_initial_position = (initialPosition == 0).astype(np.int8).sum()
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
      elif initialPosition[i][j] == 0:
        movementMatrix[i][j] = 0
      else:
        movementMatrix[i][j] = 0.001

  if debug:
    print("\nresult:")
    print(result)
    print("initial position:")
    position.print_position(initialPosition)
    print("final position:")
    position.print_position(finalPosition)
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
  np.savetxt(fileName, XY.T, fmt='%0.3f', delimiter=',')



'''
Generates and saves m traning examples
'''
def generate_and_save_m_training_examples(m, make_movement_fn=None, model_fname=None, traing_fname=None):
  if make_movement_fn == None and model_fname != None:
    modelInstance = model.load(model_fname)
    W = modelInstance['W']
    b = modelInstance['b']
    make_movement_fn = lambda x: model.predict(W, b, x)

  trainingExamples = makeTrainingExamples(make_movement_fn)

  X = trainingExamples['X']
  Y = trainingExamples['Y']

  for i in range(0, m):
    trainingExamples = makeTrainingExamples(make_movement_fn)
    X = np.append(X, trainingExamples['X'], axis = 1)
    Y = np.append(Y, trainingExamples['Y'], axis = 1)

    if len(X[0]) >= m:
      break

    if i % 100 == 0:
      print("Done: " + str(len(X[0])))

  X = X[:,0:m]
  Y = Y[:,0:m]
  
  saveTrainingExamples({
    'X': X,
    'Y': Y,
  }, fileName=traing_fname)



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



def filter_and_save_training_examples(reward, traing_fname, source_fname="m_training_examples_10000.csv"):
  raw = np.loadtxt(source_fname, delimiter=',')

  X = None
  Y = None

  for i in range(len(raw)):
    xy = raw[i].reshape(1, 18).T
    y = xy[9:18, :].astype(np.float32)
    if (y == reward).astype(np.int8).sum() == 0:
      continue

    x = xy[0:9, :].astype(np.float32)

    if X == None:
      X = x
      Y = y
    else:
      X = np.append(X, x, axis = 1)
      Y = np.append(Y, y, axis = 1)

  saveTrainingExamples({
    'X': X,
    'Y': Y,
  }, fileName=traing_fname)



def test_model(W=None, b=None, model_fname=None, show_failed_tests=False):
  if model_fname != None:
    modelInstance = model.load(model_fname)
    W = modelInstance['W']
    b = modelInstance['b']

  test_case_getters = [
    test_case_1,
    test_case_2,
    test_case_3,
    test_case_4,
    test_case_5,
    test_case_6,
  ]

  total_tests = 0
  total_passed = 0

  for i in range(len(test_case_getters)):
    test_case = test_case_getters[i]()

    passed_tests_count = 0

    for j in range(len(test_case)):
      total_tests += 1
      x = test_case[j][0:9].reshape(9, 1)
      y = test_case[j][9:18].reshape(9, 1)

      if (model.predict(W, b, x) == y).all():
        passed_tests_count += 1
        total_passed += 1
      else:
        None
        # position.printPosition(x.reshape(3, 3))

        if show_failed_tests:
          print("failed test %d in test_case_%d" % (j, i + 1))

    # print("passed tests: %d/%d" % (passed_tests_count, len(test_case)))

  print("%d%%" % (100.0 * total_passed / total_tests))

  return 1.0 * total_passed / total_tests



def test_case_1():
  return np.array([
    [ 0, 0, 0,\
      0, 0, 0,\
      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],
    [-1, 0, 0,\
      0, 0, 0,\
      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],
    [-1, 0, 0,\
     -1, 1, 0,\
      0, 1, 0,\
      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],
    [-1, 0, 0,\
      0, 1,-1,\
     -1, 1, 0,\
      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],
  ])



def test_case_2():
  return np.array([
    [-1, 0, 0,\
      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],
    [ 0, 0,-1,\
      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],
    [ 0, 0, 0,\
      0, 1, 0,\
     -1, 0, 0,\
      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],
    [ 0, 0, 0,\
      0, 1, 0,\
      0, 0,-1,\
      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],
  ])



def test_case_3():
  return np.array([
    [ 1,-1, 1,\
     -1, 1, 0,\
     -1, 1,-1,\
      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],
    [ 0,-1, 1,\
      0, 1, 0,\
      0, 0,-1,\
      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],
    [ 0, 0,-1,\
      0,-1, 0,\
      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],
    [-1, 0,-1,\
      1, 1, 0,\
      0, 0,-1,\
      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],
  ])



def test_case_4():
  return np.array([
    [ 1,-1, 1,\
      0, 0, 1,\
      0,-1,-1,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [-1, 1,-1,\
      0, 0,-1,\
      0, 1, 1,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0,-1, 1,\
      0, 0, 1,\
      0,-1,-1,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 1,-1,\
      0, 0,-1,\
      0, 0, 1,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],
  ])



def test_case_5():
  return np.array([
    [ 0,-1, 0,\
      0, 0, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 0,-1,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 0, 0,\
      0,-1, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
     -1, 0, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],
  ])



def test_case_6():
  return np.array([
    [ 0, 0,-1,\
      0, 1, 0,\
      0, 0,-1,\

      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 1, 0,\
     -1, 0,-1,\

      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0],

    [-1, 0, 0,\
      0, 1, 0,\
     -1, 0, 0,\

      0, 0, 0,\
      1, 0, 0,\
      0, 0, 0],

    [-1, 0,-1,\
      0, 1, 0,\
      0, 0, 0,\

      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],
  ])



def get_ideal_training_examples():
  return np.array([
    [ 0, 0, 0,\
      0, 0, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],



    [-1, 0, 0,\
      0, 0, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0,-1, 0,\
      0, 0, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0,-1,\
      0, 0, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 0,-1,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 0, 0,\
      0, 0,-1,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 0, 0,\
      0,-1, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 0, 0,\
     -1, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
     -1, 0, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 1, 0,\
      0, 0, 0],






    [ 0, 0, 0,\
      0,-1, 0,\
      0, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],












    [ 0, 1, 0,\
      0,-1, 0,\
      0, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0,-1, 1,\
      0, 0, 0,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0,-1, 0,\
      0, 1, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 0, 0, 0,\
      1,-1, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

























    [-1, 0, 0,\
      0, 1, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 0,-1, 0,\
      0, 1, 0,\
      0, 0, 0,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0,-1,\
      0, 1, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [ 0, 0, 0,\
      0, 1,-1,\
      0, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 0, 0, 0,\
      0, 1, 0,\
      0, 0,-1,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 1, 0,\
      0,-1, 0,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [ 0, 0, 0,\
      0, 1, 0,\
     -1, 0, 0,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
     -1, 1, 0,\
      0, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],













    [-1,-1, 0,\
      0, 1, 0,\
      0, 0, 0,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0,-1,\
      0, 1,-1,\
      0, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 0, 0, 0,\
      0, 1, 0,\
      0,-1,-1,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [ 0, 0, 0,\
     -1, 1, 0,\
     -1, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0,-1,-1,\
      0, 1, 0,\
      0, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 1,-1,\
      0, 0,-1,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 1, 0,\
     -1,-1, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [-1, 0, 0,\
     -1, 1, 0,\
      0, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],





    [ 0,-1, 0,\
     -1, 1, 0,\
      0, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0,-1, 0,\
      0, 1,-1,\
      0, 0, 0,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 0,\
      0, 1,-1,\
      0,-1, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 0, 0, 0,\
     -1, 1, 0,\
      0,-1, 0,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],









    [ 0,-1, 0,\
      0, 1, 0,\
     -1, 0, 0,\

      0, 0, 0,\
      1, 0, 0,\
      0, 0, 0],

    [-1, 0, 0,\
      0, 1,-1,\
      0, 0, 0,\

      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0,-1,\
      0, 1, 0,\
      0,-1, 0,\

      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],

    [ 0, 0, 0,\
     -1, 1, 0,\
      0, 0,-1,\

      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0],










    [ 1, 0, 0,\
      0,-1, 0,\
      0, 0,-1,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 1,\
      0,-1, 0,\
     -1, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [-1, 0, 0,\
      0,-1, 0,\
      0, 0, 1,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [ 0, 0,-1,\
      0,-1, 0,\
      1, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],






    [-1, 0,-1,\
      0, 1, 0,\
      0, 0, 1,\

      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0,-1, 1,\
      0, 1, 0,\
     -1, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 0, 0, 1,\
     -1, 1, 0,\
     -1, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0, 1,\
      0, 1, 0,\
     -1,-1, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],


    [ 0, 0,-1,\
      0, 1, 0,\
      1, 0,-1,\

      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],

    [-1, 0, 0,\
      0, 1,-1,\
      0, 0, 1,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [-1,-1, 0,\
      0, 1, 0,\
      0, 0, 1,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [-1, 0, 0,\
     -1, 1, 0,\
      0, 0, 1,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [ 1, 0, 0,\
      0, 1, 0,\
     -1, 0,-1,\

      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0],

    [ 0, 0,-1,\
      0, 1, 0,\
      1,-1, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0,-1,\
      0, 1,-1,\
      1, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 0,-1,-1,\
      0, 1, 0,\
      1, 0, 0,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [-1, 0, 1,\
      0, 1, 0,\
     -1, 0, 0,\

      0, 0, 0,\
      1, 0, 0,\
      0, 0, 0],

    [ 1, 0, 0,\
     -1, 1, 0,\
      0, 0,-1,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 1, 0, 0,\
      0, 1, 0,\
      0,-1,-1,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [ 1, 0, 0,\
      0, 1,-1,\
      0, 0,-1,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],







    [ 1,-1, 1,\
      0,-1, 0,\
      0, 0,-1,\

      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0],

    [ 0, 0, 1,\
      0,-1,-1,\
     -1, 0, 1,\

      0, 0, 0,\
      1, 0, 0,\
      0, 0, 0],

    [-1, 0, 0,\
      0,-1, 0,\
      1,-1, 1,\

      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 1, 0,-1,\
     -1,-1, 0,\
      1, 0, 0,\

      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],





    [-1, 1,-1,\
      0, 1, 0,\
      0,-1, 1,\

      0, 0, 0,\
      1, 0, 0,\
      0, 0, 0],

    [-1,-1, 1,\
      0, 1, 0,\
     -1, 0, 1,\

      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],

    [ 0,-1, 1,\
      0, 1,-1,\
     -1, 0, 1,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [-1,-1, 1,\
      0, 1,-1,\
      0, 0, 1,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [ 0, 0,-1,\
     -1, 1, 1,\
      1, 0,-1,\

      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],

    [-1, 0,-1,\
      0, 1,-1,\
      1, 0, 1,\

      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0],

    [-1, 0, 0,\
      0, 1,-1,\
      1,-1, 1,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [ 0, 0,-1,\
      0, 1,-1,\
      1,-1, 1,\

      1, 0, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 1,-1, 0,\
      0, 1, 0,\
     -1, 1,-1,\

      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],

    [ 1, 0,-1,\
      0, 1, 0,\
      1,-1,-1,\

      0, 0, 0,\
      1, 0, 0,\
      0, 0, 0],

    [ 1, 0,-1,\
     -1, 1, 0,\
      1,-1, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 1, 0, 0,\
     -1, 1, 0,\
      1,-1,-1,\

      0, 0, 1,\
      0, 0, 0,\
      0, 0, 0],

    [-1, 0, 1,\
      1, 1,-1,\
     -1, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0],

    [ 1,-1, 1,\
     -1, 1, 0,\
      0, 0,-1,\

      0, 0, 0,\
      0, 0, 0,\
      1, 0, 0],

    [ 1,-1, 1,\
     -1, 1, 0,\
     -1, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],

    [ 1, 0, 1,\
     -1, 1, 0,\
     -1, 0,-1,\

      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 1,-1, 1,\
     -1, 1, 0,\
     -1, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 0, 1],






    [ 1,-1, 1,\
     -1,-1, 0,\
      0, 1,-1,\

      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],

    [ 1,-1, 1,\
      0,-1,-1,\
      0, 1,-1,\

      0, 0, 0,\
      1, 0, 0,\
      0, 0, 0],

    [ 0,-1, 1,\
      1,-1,-1,\
     -1, 0, 1,\

      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0],

    [ 0, 0, 1,\
      1,-1,-1,\
     -1,-1, 1,\

      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],

    [-1, 1, 0,\
      0,-1,-1,\
      1,-1, 1,\

      0, 0, 0,\
      1, 0, 0,\
      0, 0, 0],

    [-1, 1, 0,\
     -1,-1, 0,\
      1,-1, 1,\

      0, 0, 0,\
      0, 0, 1,\
      0, 0, 0],


    [ 1, 0,-1,\
     -1,-1, 1,\
      1,-1, 0,\

      0, 1, 0,\
      0, 0, 0,\
      0, 0, 0],

    [ 1,-1,-1,\
     -1,-1, 1,\
      1, 0, 0,\

      0, 0, 0,\
      0, 0, 0,\
      0, 1, 0],




























    # [ 1, 0, 0,\
    #   0,-1, 0,\
    #   0, 0,-1,\

    #   0, 0, 1,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [ 0, 0, 1,\
    #   0,-1, 0,\
    #  -1, 0, 0,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 1],

    # [-1, 0, 0,\
    #   0,-1, 0,\
    #   0, 0, 1,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   1, 0, 0],

    # [ 0, 0,-1,\
    #   0,-1, 0,\
    #   1, 0, 0,\

    #   1, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 0],



    # [ 1, 0, 0,\
    #   0,-1, 0,\
    #   0, 0,-1,\

    #   0, 0, 1,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [-1, 0,-1,\
    #   0, 1, 0,\
    #   0, 0, 1,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   1, 0, 0],

    # [ 0, 0, 1,\
    #   0,-1, 0,\
    #  -1, 0, 0,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 1],

    # [ 0, 0, 0,\
    #   0,-1, 1,\
    #   0, 0,-1,\

    #   1, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [-1, 0, 0,\
    #   0,-1, 0,\
    #   0, 0, 1,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   1, 0, 0],

    # [ 0, 0, 0,\
    #   0,-1, 0,\
    #  -1, 1, 0,\

    #   0, 0, 1,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [ 0, 0,-1,\
    #   0,-1, 0,\
    #   1, 0, 0,\

    #   1, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [-1, 0, 0,\
    #   1,-1, 0,\
    #   0, 0, 0,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 1],
    
    




    # [ 1, 0, 0,\
    #   0,-1, 0,\
    #   0, 0,-1,\

    #   0, 0, 1,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [ 0, 1,-1,\
    #   0,-1, 0,\
    #   0, 0, 0,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   1, 0, 0],

    # [ 0, 0, 1,\
    #   0,-1, 0,\
    #  -1, 0, 0,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 1],

    # [ 0, 0, 0,\
    #   0,-1, 1,\
    #   0, 0,-1,\

    #   1, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [-1, 0, 0,\
    #   0,-1, 0,\
    #   0, 0, 1,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   1, 0, 0],

    # [ 0, 0, 0,\
    #   0,-1, 0,\
    #  -1, 1, 0,\

    #   0, 0, 1,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [ 0, 0,-1,\
    #   0,-1, 0,\
    #   1, 0, 0,\

    #   1, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 0],

    # [-1, 0, 0,\
    #   1,-1, 0,\
    #   0, 0, 0,\

    #   0, 0, 0,\
    #   0, 0, 0,\
    #   0, 0, 1],
  ])



def make_ideal_training_examples():
  training_examples = get_ideal_training_examples()

  X = []
  Y = []

  for i in range(len(training_examples)):
    training_example = training_examples[i]
    x = np.array(training_example[0:9].reshape(9, 1))
    y = np.array(training_example[9:18].reshape(9, 1))

    if i == 0:
      X = np.array(x)
      Y = np.array(y)
    else:
      X = np.append(X, x, axis = 1)
      Y = np.append(Y, y, axis = 1)

  Y = Y * 0.999
  Y = Y + 0.001

  saveTrainingExamples({
    'X': X,
    'Y': Y,
  }, 'ideal_training_examples.csv')
