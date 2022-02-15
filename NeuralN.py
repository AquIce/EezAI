import numpy as np
from test import *

# Our Neural Network class
class Neural_Network(object):
    def __init__(self):
        
        # The settings
        self.settings = {
          'inputSize':2, # Input layer
          'hiddenSize1':4, # Hidden layer 1
          'hiddenSize2':4, # Hidden layer 2
          'outputSize':1 # Output layer
        }

        # Our synapsis weights
        self.W1 = np.random.randn(self.settings['inputSize'], self.settings['hiddenSize1']) # Input layer - hidden layer 1 (2x3)
        self.W2 = np.random.randn(self.settings['hiddenSize1'], self.settings['hiddenSize2']) # Hidden layer 1 - hidden layer 2 (3x3)
        self.W3 = np.random.randn(self.settings['hiddenSize2'], self.settings['outputSize']) # Hidden layer 2 - output layer (3x1)


    # Forward propagation function
    def forward(self, X):

        self.z = np.dot(X, self.W1) # Matrixial multiplication beetween the inputs (X) and W1
        self.z2 = self.sigmoid(self.z) # Apply the activation (=sigmoid) function to the result (z)
        self.z3 = np.dot(self.z2, self.W2) # Matrixial multiplication beetween the hidden (z2) and W2
        output = self.sigmoid(self.z3) # Apply the activation (=sigmoid) function to the result (z3) and obtain our output
        return output

    # Activation (=sigmoid) function
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    # Activation (=sigmoid) prime function
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # Backward propagation function
    def backward(self, X, y, output):

        # Calculate the error
        self.output_error = y - output
        # Apply the sigmoid prime to this error
        self.output_delta_error = self.output_error*self.sigmoidPrime(output)

        # Calculate the hidden layer error
        self.z2_error = self.output_delta_error.dot(self.W2.T)
        # Apply the sigmoid prime to this error
        self.z2_delta_error = self.z2_error*self.sigmoidPrime(self.z2)

        # Ajust W1 weights
        self.W1 += X.T.dot(self.z2_delta_error)
        # Ajust W2 weights
        self.W2 += self.z2.T.dot(self.output_delta_error)

    # Train function
    def train(self, X, y):
            
        output = self.forward(X)
        self.backward(X, y, output)
        print("\n\n" + self.W1 + "\n\n")

    # Predict function
    def predict(self, xPrediction):
            
        print("Predicted datas after train : ")
        print("Input : \n" + str(xPrediction))
        print("Output : \n" + str(self.forward(xPrediction)))

        if(self.forward(xPrediction) < 0.5):
            print("The flower is BLUE\n")
        else:
            print("The flower is RED\n")

def getIterations():
    return 10