import numpy as np
import training
import model
import position


'''
Trains model with static tainig examples (read from file).
'''
def train_model_scenario_1(n, model_fname, training_examples_fname, m=0, alpha=0.001, iterations=10000):
  debug = False

  (W, b) = model.initialize_weights(n)
  ex = training.read_training_examples(m, fname=training_examples_fname)

  X = ex['X']
  # assert X.shape == (9, 500)

  Y = ex['Y']
  # assert Y.shape == (9, 500)


  # L is a number of NN layers
  # (L = 3 for a model 9x18x18x9)
  L = len(n) - 1
  assert len(W) == L


  for i in range(0, iterations):
    # (dW, db) = model.calcGradients(W, b, X, Y)
    (dW, db, _) = model.back_propagation(n, W, b, X, Y)

    model.update_weights(W, dW, b, db, alpha)

    if i % 300 == 0:
      print('iteration ' + str(i))
      (aL, _) = model.forward_propagation(W, b, X)
      cost = model.cost_function(Y, aL)
      # cost = model.costFunction(Y, A[L])
      # print('alpha')
      # print(alpha)
      print('cost')
      print(cost)

    if debug:
      if i > 0 and i % 3000 == 0:
        is_back_prop_correct = model.check_back_propagation(n, W, b, X, Y)

        if not is_back_prop_correct:
          print("BP is not correct")
          exit()

  print('------ end -------')

  model.save(n, W, b, model_fname)



'''
Trains model continously with generated training examples
'''
def train_model_scenario_2(n, model_fname, opponet_model_fname, alpha=0.1, iterations=5000):
  alpha0 = alpha
  decay_rate = 0.01
  modelInstance = model.load(opponet_model_fname)
  W0 = modelInstance['W']
  b0 = modelInstance['b']

  if model_fname == opponet_model_fname:
    W = W0
    b = b0
  else:
    make_movement_fn = lambda x: model.predict2(W0, b0, x)
    (W, b) = model.initialize_weights(n)

  for i in range(0, iterations):
    if model_fname == opponet_model_fname:
      make_movement_fn = lambda x: model.predict2(W, b, x)

    ex = training.make_training_examples(make_movement_fn)

    X = ex['X']
    Y = ex['Y']

    # displayTrainingExamples(X, Y)

    # (dW, db) = model.calcGradients(W, b, X, Y)
    (dW, db, _) = model.back_propagation(n, W, b, X, Y)

    alpha = alpha0 / (1 + decay_rate * i)

    model.update_weights(W, dW, b, db, alpha)

    if i % 100 == 0:
      print('iteration ' + str(i))
      (aL, _) = model.forward_propagation(W, b, X)
      cost = model.cost_function(Y, aL)
      print('cost')
      print(cost)
      print('alpha')
      print(alpha)

    # if i % 1000 == 0:
    #   is_back_prop_correct = model.checkBackPropagation(n, W, b, X, Y)

    #   if not is_back_prop_correct:
    #     print("BP is not correct")
    #     exit()

  print('------ end -------')

  model.save(n, W, b, model_fname)



'''
Trains model with static tainig examples (read from file).
'''
def train_model_scenario_3(n, model_fname, training_examples_fname, m=0, alpha=0.001, beta=0.9, iterations=10000):
  debug = False

  modelInstance = model.load(model_fname)
  W = modelInstance['W']
  b = modelInstance['b']
  # (W, b) = model.initializeWeights(n)
  vdW = np.zeros(W.shape)
  vdb = np.zeros(b.shape)
  ex = training.read_training_examples(m, fname=training_examples_fname)

  X = ex['X']
  # assert X.shape == (9, 500)

  Y = ex['Y']
  # assert Y.shape == (9, 500)


  # L is a number of NN layers
  # (L = 3 for a model 9x18x18x9)
  L = len(n) - 1
  assert len(W) == L


  for i in range(0, iterations):
    # (dW, db) = model.calcGradients(W, b, X, Y)
    (dW, db, _) = model.back_propagation(n, W, b, X, Y)

    vdW = beta * vdW + (1 - beta) * dW
    vdb = beta * vdb + (1 - beta) * db

    # model.updateWeights(W, dW, b, db, alpha)
    W = W - alpha * vdW
    b = b - alpha * vdb

    if i % 30 == 0:
      print('iteration ' + str(i))
      (aL, _) = model.forward_propagation(W, b, X)
      cost = model.cost_function(Y, aL)
      # cost = model.costFunction(Y, A[L])
      # print('alpha')
      # print(alpha)
      print('cost')
      print(cost)

    if debug:
      if i > 0 and i % 3000 == 0:
        is_back_prop_correct = model.check_back_propagation(n, W, b, X, Y)

        if not is_back_prop_correct:
          print("BP is not correct")
          exit()
    if i % 1000 == 0:
      model.save(n, W, b, model_fname)

  print('------ end -------')

  model.save(n, W, b, model_fname)



'''
Trains model continously with generated training examples
Testing with training.test_model
'''
def train_model_scenario_4(model_fname, alpha0=1, iterations=500000, beta=0.9):
  debug = False

  model_instance = model.load(model_fname)

  n = model_instance['n']
  W = model_instance['W']
  b = model_instance['b']
  
  vdW = np.zeros(W.shape)
  vdb = np.zeros(b.shape)

  decay_rate = 9.0 / iterations # it will reduce final alpha 10 times

  for i in range(0, iterations):
    if i % 1000 == 0:
      print('i: %d' % (i))

    # debug = True if i % 500 == 0 else False

    make_movement_fn = lambda x: model.predict2(W, b, x)

    ex = training.make_training_examples(make_movement_fn)

    X = ex['X']
    Y = ex['Y']

    if debug:
      print(X)
      print(Y)
      for j in range(len(X.T) - 1, -1, -1):
        x = X.T[j]
        nextPosition = position.transform_vector_into_position(x)
        x = position.transform_position_into_vector(nextPosition)

        position.print_position(nextPosition)

        (aL, _) = model.forward_propagation(W, b, x)

        print("\naL")
        print(aL)

        print('\n predicted ')

        movement = model.predict(W, b, x)
        position.print_movement(movement)

        print(' expected ')

        y = Y.T[j]
        position.print_movement(y.reshape(9, 1))

        raw_input("Press Enter to continue...")

    # displayTrainingExamples(X, Y)

    # (dW, db) = model.calcGradients(W, b, X, Y)
    (dW, db, _) = model.back_propagation(n, W, b, X, Y)

    if debug:
      if i > 0 and i % 3000 == 0:
        is_back_prop_correct = model.check_back_propagation(n, W, b, X, Y)

        if is_back_prop_correct:
          print("BP is OK")
        else:
          print("BP is not correct")
          exit()

    alpha = alpha0 / (1.0 + decay_rate * i)

    # if debug:
    if i % 1000 == 0:
      print("alpha:")
      print(alpha)

    vdW = beta * vdW + (1 - beta) * dW
    vdb = beta * vdb + (1 - beta) * db

    # model.updateWeights(W, dW, b, db, alpha)

    # W = W - alpha * vdW
    # b = b - alpha * vdb
    W = W - alpha * dW
    b = b - alpha * db
    
    if i > 0 and i % 50 == 0:
      model.save(n, W, b, model_fname)
      if debug:
        print("model saved")

  print('------ end -------')

  model.save(n, W, b, model_fname)



'''
Loads model and training examples
Then shows how the model predicts on each example
Shall be removed.
'''
def test_model_on_static_examples(model_fname, training_examples_fname, m=0):
  modelInstance = model.load(model_fname)

  # n = model9x9['n']
  W = modelInstance['W']
  b = modelInstance['b']

  trainingExamples = training.read_training_examples(m, fname=training_examples_fname)

  if m == 0:
    m = trainingExamples['X'].shape[1]

  for i in range(1000):
    # i = round(np.random.rand() * m)
    x = trainingExamples['X'].T[i]
    nextPosition = position.transform_vector_into_position(x)

    position.print_position(nextPosition)

    print(' predicted ')

    x = position.transform_position_into_vector(nextPosition)
    movement = model.predict(W, b, x)
    position.print_movement(movement)

    print(' expected ')

    y = trainingExamples['Y'].T[i]
    position.print_movement(y.reshape(9, 1))

    raw_input("Press Enter to continue...")



'''
Receives X and Y of training examples
Draws positions and movement weights.
'''
def display_training_examples(X, Y):
  XT = X.T
  YT = Y.T

  for i in range(len(XT)):
    x = XT[i]
    y = YT[i]
    display_signle_training_example(x, y)



'''
Same as display_training_examples
but for signle training example
'''
def display_signle_training_example(x, y):
  nextPosition = position.transform_vector_into_position(x)

  position.print_position(nextPosition)

  print(' expected ')

  position.print_movement(y.reshape(9, 1))

  raw_input("Press Enter to continue...")



'''
Quick visual test of a model.
'''
def test_position(model_fname):
  print('\ntest model: %s' % (model_fname))
  model_instance = model.load(model_fname)
  W = model_instance['W']
  b = model_instance['b']


  x = np.array([
    -1,-1, 0,
     0, 1, 0,
     0, 0, 0,
  ]).reshape(9, 1)

  print('\n')
  position.print_position(position.transform_vector_into_position(x))

  (aL, _) = model.forward_propagation(W, b, x)
  print("aL")
  print(aL)
  movement = model.predict(W, b, x)
  position.print_movement(movement)




# training.make_ideal_training_examples()
# train_model_scenario_3(n = [9, 27, 27, 27, 27, 9], model_fname='9x27x27x27x27x9.model', training_examples_fname='m_training_examples_3000.csv', alpha=3, iterations=1000)
# test_position('9x18x18x18x18x9.model')
# training.test_model(model_fname='9x81x81x81x9.model')

# model.play_two_games('9x81x81x9.model', '9x45x45x45x9.model')
# model.play_two_games('9x18x18x18x18x9.model', '9x81x81x9.model')
# model.play_two_games('9x81x81x9.model', '9x81x81x81x9.model')
# model.play_two_games('9x27x27x27x9.model', '9x81x81x9.model')
# model.play_two_games('9x81x81x9.model', '9x81x81x81x9.model')
# model.play_two_games('9x81x81x81x81x9.model', '9x81x81x9.model')
# training.generate_and_save_m_training_examples(100, model_fname='9x81x81x81x9.model')
# test_model_on_static_examples(model_fname='9x27x27x27x27x9.model', training_examples_fname='m_training_examples.csv')
# model.create([9, 27, 27, 9], '9x27x27x9.model')
# train_model_scenario_4(model_fname='9x27x27x9.model', alpha0=0.1)
test_position('9x27x27x9.model')
# model.play_two_games('9x27x27x9.model', '9x81x81x9.model')

def spy_on_training_process(model_fname):
  model_instance = model.load(model_fname)

  n = model_instance['n']
  W = model_instance['W']
  b = model_instance['b']
  
  vdW = np.zeros(W.shape)
  vdb = np.zeros(b.shape)

  alpha0 = 0.3
  beta=0.9
  iterations = 100000
  decay_rate = 4.0 / iterations

  pos_trains = 0
  x_to_investigate = None

  for i in range(0, iterations):
    make_movement_fn = lambda x: model.predict2(W, b, x)

    ex = make_training_examples(make_movement_fn)

    X = ex['X']
    Y = ex['Y']

    x = X[:, 0].reshape(9, 1)

    if x_to_investigate == None:
      x_to_investigate = x
      position_to_investigate = x_to_investigate.reshape(3, 3)

    if (x == x_to_investigate).all():
      position.print_position(position_to_investigate)
      (_, _, aLbefore) = model.predict3(W, b, x_to_investigate)

    (dW, db, _) = model.back_propagation(n, W, b, X, Y)

    alpha = alpha0 / (1.0 + decay_rate * i)

    vdW = beta * vdW + (1 - beta) * dW
    vdb = beta * vdb + (1 - beta) * db

    W = W - alpha * dW
    b = b - alpha * db

    if i > 0 and i % 1000 == 0:
      model.save(n, W, b, model_fname)
      print('========saved=======')

    if (x == x_to_investigate).all():
      pos_trains += 1
      position.print_position(position_to_investigate)
      (_, _, aLafter) = model.predict3(W, b, x_to_investigate)
      print('\niteration: %d, position trained times: %d' % (i, pos_trains))
      y = Y[:, 0].reshape(9, 1)
      table = np.concatenate((aLbefore, aLafter, y), axis = 1)
      print(table)

def make_training_examples(make_movement_fn):
  random = np.random.rand()

  chance_of_custom_case = random < 0.25

  if not chance_of_custom_case:
    return training.make_training_examples(make_movement_fn)

  non_zero_position = np.array([
    0, 0, 0,
    1, 1, 0,
    0, 0,-1,
  ]).reshape(3, 3)

  training_examples = training.make_training_examples_rec(non_zero_position, make_movement_fn)

  return {
    'X': training_examples['X'],
    'Y': training_examples['Y'],
  }

# spy_on_training_process('9x81x81x9.model')
