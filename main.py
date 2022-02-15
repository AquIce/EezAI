from NeuralN import *
from lib import *

NN = Neural_Network

# Input values :
x_input = [[3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1], [4,1.5]]

# Output values (1 = red, 0 = blue)
y = [[1], [0], [1], [0], [1], [0], [1], [0]]

# Put all the input values beetween 0 amd 1
x_input = MatrixEchelon(x_input, 0)

# Put all the known output's input in X (to train the function)
X = MatrixReshape(x_input, 8, True)

# The input we don't know the output
xPrediction = MatrixReshape(x_input, 8, False)

NN = Neural_Network()

# Choose a number of iterations
iter = getIterations()

# Create a void list to contain the input, real and predicted datas
data_list = [0] * iter

# Make this iter times
for i in range(iter):
    # Create a variable who contains predicted datas (using the forward function) and round it to 2 decimals
    predicted = MatrixRound(NN.forward(X),2)

    # Print the actual i value (ex : #32)
    print("# " + str(i) + "\n")
    # Print the input values (=[3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1])
    print("Input values : \n" + str(X))
    # Print the real output ([1], [0], [1], [0], [1], [0], [1], [0])
    print("Real output : \n" + str(y))
    # Print the predicted output (8 values)
    print("Predicted output : \n" + str(predicted) + "\n")

    # Create a dictionnary who contains all the input, real and predicted datas for this iteration
    datas_dict = {
        "Input": X,
        "Real": y, 
        "Predicted": predicted
    }

    # Add it to data_list
    data_list[i] = datas_dict

    # Train our Neural Network
    NN.train(X,y)

# Predict the last output (has to be red = 0)
NN.predict(xPrediction)

# Ask the user if he want to print datas
yorn = input("Print datas ? ")

if yorn == 'y':
    # Choose the data to print
    data_to_print = int(input("Which data ?"))
    print(data_list[data_to_print])
    
# Save all the personnal datas in log.npy
np.save('log.npy', data_list)

# Print the datas contained in log.py
print(np.load('log.npy', allow_pickle='TRUE'))