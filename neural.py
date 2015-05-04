import random
import operator

class Perceptron:


    def __init__(self, number_inputs, learning_rate=0.01):
        self.weights = [random.uniform(-1,1) for n in range(number_inputs)]
        self.learning = learning_rate

    def activation(self, i):
        """
        Activation energy to fire the Perceptron signal.
        """
        if i > 0: 
            return 1
        else:
            return -1
    
    def feedforward(self, inputs):
        """
        Inputs fed forward into the next Perceptron as an weighted activation output. 
        """
        weighted_inputs = map(operator.mul, inputs, self.weights)
        total = sum(weighted_inputs)

        return self.activation(total)

    def train(self, test_point):
        """
        Trains the perception based on supervised learning data item.
        """
        output = test_point.pop()
        inputs = test_point

        guess = self.feedforward(inputs)
        error =  output - guess
        updates = [self.learning*error*i for i in inputs]
        
        return map(operator.add, self.weights, updates)

    def training(self, tests):
        """
        Given a set of test examples to train weights on.
        """
        for test in tests:
            self.weights = self.train(test)

    def __str__(self):
        """
        Prints formated perception.
        """
        return ('weights:' + str(self.weights) + '\n' + 
                'learning rate:' + str(self.learning) + '\n')

