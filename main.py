import numpy as np
import training
import model
import position


'''
Trains model with static tainig examples (read from file).
'''
def trainModelScenario1(n, model_fname, training_examples_fname, m=0, alpha=0.001, iterations=10000):
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
def trainModelScenario2(n, model_fname, opponet_model_fname, alpha=0.1, iterations=5000):
  alpha0 = alpha
  decay_rate = 0.01
  modelInstance = model.load(opponet_model_fname)
  W0 = modelInstance['W']
  b0 = modelInstance['b']

  if model_fname == opponet_model_fname:
    W = W0
    b = b0
  else:
    make_movement_fn = lambda x: model.predict(W0, b0, x)
    (W, b) = model.initialize_weights(n)

  for i in range(0, iterations):
    if model_fname == opponet_model_fname:
      make_movement_fn = lambda x: model.predict(W, b, x)

    ex = training.makeTrainingExamples(make_movement_fn)

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
def trainModelScenario3(n, model_fname, training_examples_fname, m=0, alpha=0.001, beta=0.9, iterations=10000):
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
def trainModelScenario4(model_fname, opponet_model_fname, alpha0=0.1, iterations=50000, beta=0.9):
  debug = False

  model_instance = model.load(model_fname)

  n = model_instance['n']
  W = model_instance['W']
  b = model_instance['b']
  
  vdW = np.zeros(W.shape)
  vdb = np.zeros(b.shape)

  if model_fname == opponet_model_fname:
    make_movement_fn = None
  else:
    omi = model.load(model_fname)
    make_movement_fn = lambda x: model.predict(omi['W'], omi['b'], x)

  decay_rate = 4.0 / iterations

  for i in range(0, iterations):
    # debug = True if i % 500 == 0 else False

    if make_movement_fn == None:
      make_movement_fn = lambda x: model.predict(W, b, x)

    ex = training.makeTrainingExamples(make_movement_fn)

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
        nextPosition = position.positionFromVector(x)
        x = position.reshapePositionInVector(nextPosition)

        position.printPosition(nextPosition)

        (aL, _) = model.forward_propagation(W, b, x)

        print("\naL")
        print(aL)

        print('\n predicted ')

        movement = model.predict(W, b, x)
        position.printMovement(movement)

        print(' expected ')

        y = Y.T[j]
        position.printMovement(y.reshape(9, 1))

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
                position.printPosition(x.reshape(3, 3))

        if False:
          for j in range(len(X.T)):
            x = X.T[j]
            nextPosition = position.positionFromVector(x)

            position.printPosition(nextPosition)

            print(' predicted before')

            x = position.reshapePositionInVector(nextPosition)
            movement = model.predict(Wprev, bprev, x)
            position.printMovement(movement)

            print(' predicted after')

            x = position.reshapePositionInVector(nextPosition)
            movement = model.predict(W, b, x)
            position.printMovement(movement)

            print(' expected ')

            y = Y.T[j]
            position.printMovement(y.reshape(9, 1))

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



def testModel(model_fname, training_examples_fname, m=0):
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
    nextPosition = position.positionFromVector(x)

    position.printPosition(nextPosition)

    print(' predicted ')

    x = position.reshapePositionInVector(nextPosition)
    movement = model.predict(W, b, x)
    position.printMovement(movement)

    print(' expected ')

    y = trainingExamples['Y'].T[i]
    position.printMovement(y.reshape(9, 1))

    raw_input("Press Enter to continue...")



def displayTrainingExamples(X, Y):
  XT = X.T
  YT = Y.T

  for i in range(len(XT)):
    x = XT[i]
    y = YT[i]
    displayTrainingExample(x, y)



def displayTrainingExample(x, y):
  nextPosition = position.positionFromVector(x)

  position.printPosition(nextPosition)

  print(' expected ')

  position.printMovement(y.reshape(9, 1))

  raw_input("Press Enter to continue...")




# trainModelScenario1(n = [9, 81, 81, 81, 9], model_fname='9x81x81x81x9.model', training_examples_fname='m_training_examples_1250.csv', alpha=0.3, iterations=3000)

# trainModelScenario(n = [9, 18, 18, 9], model_fname = '9x18x18x9.model', training_examples_fname='m_training_examples_24.csv', alpha = 3, iterations=1000)
# trainModelScenario(n = [9, 18, 9], model_fname = '9x18x9.model', training_examples_fname='m_training_examples_24.csv', alpha = 3, iterations=30)
# trainModelScenario(n = [9, 9], model_fname = '9x9.model', training_examples_fname='m_training_examples_24.csv', alpha = 3, iterations=1000)
# trainModelScenario2(n = [9, 27, 27, 27, 9], model_fname='9x27x27x27x9.model', opponet_model_fname='9x27x27x27x9.model')
# trainModelScenario2(n = [9, 81, 81, 81, 9], model_fname='9x81x81x81x9.model', opponet_model_fname='9x45x45x45x9.model', iterations=1000, alpha=3)
# # # trainModelScenario2(n = [9, 81, 81, 81, 81, 9], model_fname='9x81x81x81x81x9.model', opponet_model_fname='9x81x81x81x81x9.model', iterations=1000, alpha=1)


# testModel(model_fname='9x9.model', training_examples_fname='m_training_examples_24.csv')
# testModel(model_fname='9x18x9.model', training_examples_fname='m_training_examples.csv')
# testModel(model_fname='9x18x18x9.model', training_examples_fname='m_training_examples.csv')
# testModel(model_fname='9x27x27x9.model', training_examples_fname='m_training_examples.csv')
# testModel(model_fname='9x27x27x27x9.model', training_examples_fname='m_training_examples.csv')
# testModel(model_fname='9x81x81x81x9.model', training_examples_fname='m_training_examples_2500.csv')
# testModel(model_fname='9x27x27x27x9.model', training_examples_fname='m_training_examples_500.csv')
# testModel(model_fname='9x36x36x36x9.model', training_examples_fname='m_training_examples_1250.csv')
# testModel('9x9x9x9.model')
# testModel('9x81x81x9.model')

# training.generate_and_save_m_training_examples(10000, traing_fname="m_training_examples_10000.csv")
# trainModelScenario1(n = [9, 81, 81, 81, 9], model_fname='9x81x81x81x9.model', training_examples_fname='training_examples_100_percent.csv', alpha=1, iterations=3000)
# trainModelScenario2(n = [9, 81, 81, 81, 9], model_fname='9x81x81x81x9.model', opponet_model_fname='9x81x81x81x9.model', iterations=2000, alpha=0.3)
# trainModelScenario3(n = [9, 81, 81, 81, 9], model_fname='9x81x81x81x9.model', training_examples_fname='training_examples_100_percent.csv', alpha=1, iterations=1200)
# testModel(model_fname='9x81x81x81x9.model', training_examples_fname='training_examples_25_percent.csv')
# testModel(model_fname='9x81x81x81x9.model', training_examples_fname='training_examples_13_percent.csv')

# training.filter_and_save_training_examples(reward=0.5, traing_fname="training_examples_50_percent.csv")
# training.filter_and_save_training_examples(reward=0.25, traing_fname="training_examples_25_percent.csv")
# training.filter_and_save_training_examples(reward=0.125, traing_fname="training_examples_13_percent.csv")

# testModel(model_fname='9x81x81x81x9.model', training_examples_fname='m_training_examples_400.csv')

# trainModelScenario4(n = [9, 81, 81, 81, 9], model_fname='9x81x81x81x9.model', opponet_model_fname='9x81x81x81x9.model', iterations=1000, alpha=0.3)
# training.generate_and_save_m_training_examples(240, model_fname='9x81x81x81x81x81x81x9.model', traing_fname="m_training_examples_240.csv")
# testModel(model_fname='9x81x81x81x81x81x81x9.model', training_examples_fname='m_training_examples_240.csv')

# training.test_model(model_fname='9x81x81x81x9.model')





def foo():
  modelInstance = model.load('9x81x81x81x81x81x81x9.model')

  n = modelInstance['n']
  n = [9, 81, 81, 81, 81, 81, 81, 9]
  # (W, b) = model.initializeWeights(n)
  W = modelInstance['W']
  b = modelInstance['b']

  test_case = training.test_case_1()
  # print(test_case)

  for i in range(4):
    x = test_case[i, 0:9].reshape(9, 1)
    y = test_case[i, 9:18].reshape(9, 1)
    position.printPosition(position.positionFromVector(x))
    position.printMovement(y)

    X = np.array(x)
    Y = np.array(y)
    (aL1, _) = model.forward_propagation(W, b, X)
    (dW, db, _) = model.back_propagation(n, W, b, X, Y)

    alpha = 0.03

    for j in range(len(W)):
      W[j] = W[j] - alpha * dW[j]
      b[j] = b[j] - alpha * db[j]

    (aL2, _) = model.forward_propagation(W, b, X)
    print("al1/al2 results:")
    print(np.concatenate((aL1, aL2, np.round((aL2 - aL1) * 1000)), axis=1))

    raw_input("...")



# trainModelScenario4(n = [9, 9, 9], model_fname='9x9x9.model', opponet_model_fname='9x9x9.model', iterations=3000, alpha=1)
# foo()
# training.generate_and_save_m_training_examples(3000, traing_fname="m_training_examples_3000.csv", model_fname='9x45x45x45x45x9.model')
# testModel(model_fname='9x45x45x45x45x9.model', training_examples_fname='m_training_examples_3000.csv')
# training.test_model(model_fname='9x45x45x45x45x9.model')


# trainModelScenario3(n = [9, 45, 45, 45, 45, 9], model_fname='9x45x45x45x45x9.model', training_examples_fname='m_training_examples_30000.csv', alpha=3, iterations=10000)
# trainModelScenario4(model_fname='9x27x27x27x9.model', opponet_model_fname='9x27x27x27x9.model', iterations=300000, alpha0=30)

def create_model(n, model_fname):
  # n = [9, 81, 81, 81, 81, 81, 81, 9]
  (W, b) = model.initialize_weights(n)
  model.save(n, W, b, model_fname)

# create_model([9, 27, 27, 27, 27, 9], '9x27x27x27x27x9.model')
# create_model([9, 81, 81, 81, 81, 81, 81, 9], '9x81x81x81x81x81x81x9.model')

def how_we_update_w_b(model_fname):
  model_instance = model.load(model_fname)
  n = model_instance['n']
  W = model_instance['W']
  b = model_instance['b']

  # make_movement_fn = lambda x: model.predict(W, b, x)
  # ex = training.makeTrainingExamples(make_movement_fn)

  # X = ex['X']
  # Y = ex['Y']

  # print(X)
  # print(Y)

  # training.saveTrainingExamples({
  #   'X': X,
  #   'Y': Y,
  # }, fileName='how_we_ex.csv')

  ex = training.read_training_examples(0, 'how_we_ex.csv')

  X = ex['X']
  Y = ex['Y']

  x = X.T[8].reshape(9, 1)
  y = Y.T[8].reshape(9, 1)

  # print(x)
  print(y)

  position.printPosition(position.positionFromVector(x))

  (aL, _) = model.forward_propagation(W, b, x)
  movement = model.predict(W, b, x)

  print('aL')
  print(aL)
  print('movement')
  print(movement)

  # alpha = 3
  # iterations = 10000

  # for _ in range(iterations):
  #   (dW, db, _) = model.back_propagation(n, W, b, X, Y)

  #   W = W - alpha * dW
  #   b = b - alpha * db

  # (aL, _) = model.forward_propagation(W, b, x)
  # movement = model.predict(W, b, x)

  # print('aL')
  # print(aL)
  # print('movement')
  # print(movement)


# how_we_update_w_b('9x27x27x27x9.model')

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
  position.printPosition(position.positionFromVector(x))

  (aL, _) = model.forward_propagation(W, b, x)
  print("aL")
  print(aL)
  movement = model.predict(W, b, x)
  position.printMovement(movement)


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
  position.printPosition(position.positionFromVector(x))

  (aL, _) = model.forward_propagation(W, b, x)
  print("aL")
  print(aL)
  movement = model.predict(W, b, x)
  position.printMovement(movement)



# training.make_ideal_training_examples()
# create_model([9, 18, 18, 18, 18, 9], '9x18x18x18x18x9.model')
# trainModelScenario3(n = [9, 27, 27, 27, 27, 9], model_fname='9x27x27x27x27x9.model', training_examples_fname='m_training_examples_3000.csv', alpha=3, iterations=1000)
# trainModelScenario4(model_fname='9x18x18x18x18x9.model', opponet_model_fname='9x18x18x18x18x9.model', iterations=300000, alpha0=1)
# test_position('9x18x18x18x18x9.model')
# training.test_model(model_fname='9x81x81x81x9.model')

model.play_the_game('9x18x18x18x18x9.model', '9x36x36x36x36x9.model')
