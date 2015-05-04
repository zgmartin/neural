import unittest
import neural

class PerceptronTest(unittest.TestCase):

    def setUp(self):
        self.perceptron = neural.Perceptron(2)
        self.perceptron.weights = [.5,-.5]

    def activation_test(self):
        answer = -1
        result = self.perceptron.activation(0)
        self.assertEquals(answer,result)

    def feedforward_test(self):
        #negative case
        answer = -1
        result = self.perceptron.feedforward([1,2])
        self.assertEquals(answer,result)
        #positive case
        answer = 1
        result = self.perceptron.feedforward([2,1])
        self.assertEquals(answer,result)

    def train_test(self):
        answer = [.52, -.48]
        test_point = [1,1,1]
        result = self.perceptron.train(test_point)
        self.assertEquals(answer,result)
