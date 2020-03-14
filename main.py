import numpy as np
import training


startPosition = np.zeros((3, 3))

trainingExamples = training.makeTrainingExampleRec(startPosition)


print('training examples X')
print(trainingExamples['X'])

print('training examples Y')
print(trainingExamples['Y'])
