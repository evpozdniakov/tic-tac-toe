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
def train_model_scenario_4(model_fname, alpha0=0.1, iterations=50000, beta=0.9):
  debug = False

  model_instance = model.load(model_fname)

  n = model_instance['n']
  W = model_instance['W']
  b = model_instance['b']
  
  vdW = np.zeros(W.shape)
  vdb = np.zeros(b.shape)

  decay_rate = 4.0 / iterations

  for i in range(0, iterations):
    # debug = True if i % 500 == 0 else False

    make_movement_fn = lambda x: model.predict2(W, b, x)

    ex = training.make_training_examples(make_movement_fn)

    X = ex['X']
    Y = ex['Y']

    # test_case = training.test_case_3()
    # # print(test_case)

    # x = test_case[3, 0:9].reshape(9, 1)
    # y = test_case[3, 9:18].reshape(9, 1)

    # X = np.array(x)
    # Y = np.array(y)

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
    if i % 100 == 0:
      print("alpha:")
      print(alpha)

    vdW = beta * vdW + (1 - beta) * dW
    vdb = beta * vdb + (1 - beta) * db

    # model.updateWeights(W, dW, b, db, alpha)
    if debug:
      test_result1 = training.test_model(W, b)

    # W = W - alpha * vdW
    # b = b - alpha * vdb
    W = W - alpha * dW
    b = b - alpha * db

    if debug:
      test_result2 = training.test_model(W, b)

      if test_result2 < test_result1:
        print("\n DROP FROM %02f to %02f" % (test_result1, test_result2))
        Wprev = W + alpha * vdW
        bprev = b + alpha * vdb

        test_case_getters = [
          training.test_case_1,
          training.test_case_2,
          training.test_case_3,
          training.test_case_4,
          training.test_case_5,
          training.test_case_6,
        ]

        for i in range(len(test_case_getters)):
          test_case = test_case_getters[i]()

          for j in range(len(test_case)):
            x = test_case[j][0:9].reshape(9, 1)
            y = test_case[j][9:18].reshape(9, 1)

            if (model.predict(Wprev, bprev, x) == y).all():
              if not (model.predict(W, b, x) == y).all():
                print("\nfailed test position:")
                position.print_position(x.reshape(3, 3))

        if False:
          for j in range(len(X.T)):
            x = X.T[j]
            nextPosition = position.transform_vector_into_position(x)

            position.print_position(nextPosition)

            print(' predicted before')

            x = position.transform_position_into_vector(nextPosition)
            movement = model.predict(Wprev, bprev, x)
            position.print_movement(movement)

            print(' predicted after')

            x = position.transform_position_into_vector(nextPosition)
            movement = model.predict(W, b, x)
            position.print_movement(movement)

            print(' expected ')

            y = Y.T[j]
            position.print_movement(y.reshape(9, 1))

            raw_input("Press Enter to continue...")

    # for k in range(len(X.T)):
    #   x = X[:, k]
    #   y = Y[:, k].reshape(9, 1)
    #   aL1 = AL1[:, k].reshape(9, 1)
    #   aL2 = AL2[:, k].reshape(9, 1)

    #   received_position = position.positionFromVector(x)
    #   print("received position:")
    #   position.printPosition(received_position)

    #   print("y:")
    #   position.printMovement(y)

    #   print("al1/al2 results:")
    #   print(np.concatenate((aL1, aL2, aL2 - aL1), axis=1))

    #   raw_input("...")


    if debug:
      test_result = training.test_model(W, b)
      if test_result > 0.65:
        print("test_result: %02f" % (test_result))

      if test_result == 1:
        print("------ all tests pass ------")
        break

      if i % 50 == 0:
        print("\n" + str(i))
        print(test_result)
    
    if i > 0 and i % 100 == 0:
      model.save(n, W, b, model_fname)
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
  #  -1,-1, 0,\
  #   0, 1, 0,\
  #   0, 0, 1,\
    0,-1, 0,\
    0, 1, 0,\
    0, 0, 0,\
  ]).reshape(9, 1)

  print('\n')
  position.print_position(position.transform_vector_into_position(x))

  (aL, _) = model.forward_propagation(W, b, x)
  print("aL")
  print(aL)
  movement = model.predict(W, b, x)
  position.print_movement(movement)


  x = np.array([
    # 0, 0, 0,\
    # 0, 0, 0,\
    # 0, 0, 0,\
    0, 0, 0,\
    0,-1, 0,\
    0, 0, 0,\
  #   1, 0, 0,\
  #   0, 1,-1,\
  #  -1, 1,-1,\
  #  -1, 0, 0,\
  #   0,-1, 0,\
  #   1,-1, 1,\
  ]).reshape(9, 1)

  print('\n')
  position.print_position(position.transform_vector_into_position(x))

  (aL, _) = model.forward_propagation(W, b, x)
  print("aL")
  print(aL)
  movement = model.predict(W, b, x)
  position.print_movement(movement)



# training.make_ideal_training_examples()
# model.create([9, 18, 18, 18, 18, 9], '9x18x18x18x18x9.model')
# train_model_scenario_3(n = [9, 27, 27, 27, 27, 9], model_fname='9x27x27x27x27x9.model', training_examples_fname='m_training_examples_3000.csv', alpha=3, iterations=1000)
# train_model_scenario_4(model_fname='9x18x18x18x18x9.model', iterations=300000, alpha0=1)
# test_position('9x18x18x18x18x9.model')
# training.test_model(model_fname='9x81x81x81x9.model')

# # # model.play_two_games('9x18x18x18x18x9.model', '9x81x81x81x9.model')
# model.play_two_games('9x18x18x18x18x9.model', '9x45x45x45x9.model')
# model.play_two_games('9x27x27x27x9.model', '9x81x81x81x9.model')
# model.play_two_games('9x81x81x81x81x9.model', '9x81x81x81x9.model')
training.generate_and_save_m_training_examples(100, model_fname='9x81x81x81x9.model')