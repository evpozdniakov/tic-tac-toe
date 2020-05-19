import numpy as np
import training
import position

class TestMakeSingleTrainingExampleForMainPlayer:
  def test_non_final_player_x(self):
    position_before = np.array([
      0,-1, 0,
      0, 1, 0,
      0, 0, 0,
    ]).reshape(3, 3)

    result_position = np.array([
      1,-1, 0,
      0, 1, 0,
      0, 0, 0,
    ]).reshape(3, 3)

    movement = {
      'coords': (0, 0),
      'highest_al': 0.4,
      'result_position': result_position,
    }

    highest_al = 0.5

    final_position = np.array([
      1,-1, 0,
      0, 1,-1,
      0, 0, 1,
    ]).reshape(3, 3)

    (x, y, al) = training.make_single_training_example_for_main_player(position_before, movement, final_position, highest_al)

    expected_x = np.array([
      0,-1, 0,
      0, 1, 0,
      0, 0, 0,
    ]).reshape(9, 1)

    # test x
    assert (x == expected_x).all()

    expected_y = np.array([
      0.5, 0.001, 0,
      0, 0.001, 0,
      0, 0, 0,
    ]).reshape(9, 1)

    # test y
    assert (y == expected_y).all()

    # test al
    assert al == 0.4

  def test_non_final_player_o(self):
    position_before = np.array([
      1,-1, 0,
      0, 1, 0,
      0, 0, 0,
    ]).reshape(3, 3)

    result_position = np.array([
      1,-1, 0,
      0, 1, 0,
      0, 0,-1,
    ]).reshape(3, 3)

    movement = {
      'coords': (2, 2),
      'highest_al': 0.2,
      'result_position': result_position,
    }

    highest_al = 0.3

    final_position = np.array([
      1,-1, 0,
      1, 1,-1,
      1, 0,-1,
    ]).reshape(3, 3)

    (x, y, al) = training.make_single_training_example_for_main_player(position_before, movement, final_position, highest_al)

    expected_x = np.array([
     -1, 1, 0,
      0,-1, 0,
      0, 0, 0,
    ]).reshape(9, 1)

    # test x
    assert (x == expected_x).all()

    expected_y = np.array([
      0.001, 0.001, 0,
      0, 0.001, 0,
      0, 0, 0.3,
    ]).reshape(9, 1)

    # test y
    assert (y == expected_y).all()

    # test al
    assert al == 0.2

    # test al
    assert al == 0.2

  def test_player_x_wins(self):
    position_before = np.array([
      1,-1, 0,
      0, 1,-1,
      0, 0, 0,
    ]).reshape(3, 3)

    result_position = np.array([
      1,-1, 0,
      0, 1,-1,
      0, 0, 1,
    ]).reshape(3, 3)

    movement = {
      'coords': (2, 2),
      'highest_al': 0.6,
      'result_position': result_position,
    }

    highest_al = 0.7

    final_position = position.clone(result_position)

    (x, y, al) = training.make_single_training_example_for_main_player(position_before, movement, final_position, highest_al)

    expected_x = np.array([
      1,-1, 0,
      0, 1,-1,
      0, 0, 0,
    ]).reshape(9, 1)

    # test x
    assert (x == expected_x).all()

    expected_y = np.array([
      0.001, 0.001, 0,
      0, 0.001, 0.001,
      0, 0, 1,
    ]).reshape(9, 1)

    # test y
    assert (y == expected_y).all()

    # test al
    assert al == 0.6

  def test_player_x_makes_draw(self):
    position_before = np.array([
      1,-1, 1,
     -1, 1, 0,
     -1, 1,-1,
    ]).reshape(3, 3)

    result_position = np.array([
      1,-1, 1,
     -1, 1, 1,
     -1, 1,-1,
    ]).reshape(3, 3)

    movement = {
      'coords': (1, 2),
      'highest_al': 0.3,
      'result_position': result_position,
    }

    highest_al = 0.4

    final_position = position.clone(result_position)

    (x, y, al) = training.make_single_training_example_for_main_player(position_before, movement, final_position, highest_al)

    expected_x = np.array([
      1,-1, 1,
     -1, 1, 0,
     -1, 1,-1,
    ]).reshape(9, 1)

    # test x
    assert (x == expected_x).all()

    expected_y = np.array([
      0.001, 0.001, 0.001,
      0.001, 0.001, 0.5,
      0.001, 0.001, 0.001,
    ]).reshape(9, 1)

    # test y
    assert (y == expected_y).all()

    # test al
    assert al == 0.3

  def test_player_x_looses(self):
    position_before = np.array([
      1, 0,-1,
     -1, 1, 0,
      0, 1,-1,
    ]).reshape(3, 3)

    result_position = np.array([
      1, 0,-1,
     -1, 1, 0,
      1, 1,-1,
    ]).reshape(3, 3)

    movement = {
      'coords': (2, 0),
      'highest_al': 0.1,
      'result_position': result_position,
    }

    highest_al = 0.2

    final_position = np.array([
      1, 0,-1,
     -1, 1,-1,
      1, 1,-1,
    ]).reshape(3, 3)

    (x, y, al) = training.make_single_training_example_for_main_player(position_before, movement, final_position, highest_al)

    expected_x = np.array([
      1, 0,-1,
     -1, 1, 0,
      0, 1,-1,
    ]).reshape(9, 1)

    # test x
    assert (x == expected_x).all()

    expected_y = np.array([
      0.001, 0, 0.001,
      0.001, 0.001, 0,
      0.1, 0.001, 0.001,
    ]).reshape(9, 1)

    # test y
    assert (y == expected_y).all()

    # test al
    assert al == 0.1

  def test_player_o_wins(self):
    position_before = np.array([
      1, 0,-1,
     -1, 1, 0,
      1, 1,-1,
    ]).reshape(3, 3)

    result_position = np.array([
      1, 0,-1,
     -1, 1,-1,
      1, 1,-1,
    ]).reshape(3, 3)

    movement = {
      'coords': (1, 2),
      'highest_al': 0.6,
      'result_position': result_position,
    }

    highest_al = 0.7

    final_position = position.clone(result_position)

    (x, y, al) = training.make_single_training_example_for_main_player(position_before, movement, final_position, highest_al)

    expected_x = np.array([
     -1, 0, 1,
      1,-1, 0,
     -1,-1, 1,
    ]).reshape(9, 1)

    # test x
    assert (x == expected_x).all()

    expected_y = np.array([
      0.001, 0, 0.001,
      0.001, 0.001, 1,
      0.001, 0.001, 0.001,
    ]).reshape(9, 1)

    # test y
    assert (y == expected_y).all()

    # test al
    assert al == 0.6

  def test_player_o_makes_draw(self):
    position_before = np.array([
      1, 0, 1,
     -1, 1, 0,
     -1, 1,-1,
    ]).reshape(3, 3)

    result_position = np.array([
      1,-1, 1,
     -1, 1, 0,
     -1, 1,-1,
    ]).reshape(3, 3)

    movement = {
      'coords': (0, 1),
      'highest_al': 0.3,
      'result_position': result_position,
    }

    highest_al = 0.4

    # final_position = position.clone(result_position)
    final_position = np.array([
      1,-1, 1,
     -1, 1, 1,
     -1, 1,-1,
    ]).reshape(3, 3)

    (x, y, al) = training.make_single_training_example_for_main_player(position_before, movement, final_position, highest_al)

    expected_x = np.array([
     -1, 0,-1,
      1,-1, 0,
      1,-1, 1,
    ]).reshape(9, 1)

    # test x
    assert (x == expected_x).all()

    expected_y = np.array([
      0.001, 0.5, 0.001,
      0.001, 0.001, 0,
      0.001, 0.001, 0.001,
    ]).reshape(9, 1)

    # test y
    assert (y == expected_y).all()

    # test al
    assert al == 0.3

  def test_player_o_looses(self):
    position_before = np.array([
      1,-1, 0,
      0, 1, 0,
      0, 0, 0,
    ]).reshape(3, 3)

    result_position = np.array([
      1,-1, 0,
     -1, 1, 0,
      0, 0, 0,
    ]).reshape(3, 3)

    movement = {
      'coords': (1, 0),
      'highest_al': 0.3,
      'result_position': result_position,
    }

    highest_al = 0.4

    # final_position = position.clone(result_position)
    final_position = np.array([
      1,-1, 0,
     -1, 1, 0,
      0, 0, 1,
    ]).reshape(3, 3)

    (x, y, al) = training.make_single_training_example_for_main_player(position_before, movement, final_position, highest_al)

    expected_x = np.array([
     -1, 1, 0,
      0,-1, 0,
      0, 0, 0,
    ]).reshape(9, 1)

    # test x
    assert (x == expected_x).all()

    expected_y = np.array([
      0.001, 0.001, 0,
      0.1, 0.001, 0,
      0, 0, 0,
    ]).reshape(9, 1)

    # test y
    assert (y == expected_y).all()

    # test al
    assert al == 0.3
