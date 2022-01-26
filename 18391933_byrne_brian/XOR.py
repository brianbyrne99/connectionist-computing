import numpy as np
from MLP import MLP


def XOR(max_epochs, learning_rate, xor_file):
    # intialising MultiLayer perceptron for XOR function
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_output = np.array([[0], [1], [1], [0]])
    num_inputs = 2
    num_hidden_units = 4
    num_outputs =1
    mlp = MLP(num_inputs, num_hidden_units, num_outputs)
    mlp.randomise()

    #Writing results to a log file so we can perform a deeper statistical analysis later on
    xor_file.write('MLP structure for XOR : #inputs:{0}   #Hidden Units:{1}   #Outputs:{2}\n'.format(num_inputs, num_hidden_units, num_outputs))
    xor_file.write('Maximum number of Epochs:{0}\n'.format(max_epochs))
    xor_file.write('Learning rate:{0}\n'.format(learning_rate))

    print('MLP structure for XOR : #inputs:{0}   #Hidden Units:{1}   #Outputs:{2}\n'.format(num_inputs, num_hidden_units, num_outputs))
    print('Maximum number of Epochs:{0}\n'.format(max_epochs))
    print('Learning rate:{0}\n'.format(learning_rate))

    # Testing before training
    xor_file.write("Before Training Analysis")
    print("Before Training Analysis")

    index = 0
    for arr in inputs:
        mlp.forward(arr, 'xor')
        xor_file.write('{0}Expected: {1}\tOutput: {2}\n'.format(arr, expected_output[index], mlp.O))
        print('{0}Expected: {1}\tOutput: {2}\n'.format(arr, expected_output[index], mlp.O))
        index +=1
    
    #Training
    xor_file.write("Training the Perceptron\n")
    print("Training the Perceptron\n")
    for i in range(0, max_epochs):
        mlp.forward(inputs, 'xor')
        error_rate = 0
        error_rate = mlp.backwards(inputs, expected_output, 'xor')
        # update the weights with learning weight
        mlp.updateWeights(learning_rate)


        if (i + 1) % (max_epochs / 20) == 0:
            xor_file.write('Number of Epochs: {0}\tError Rate: {1}\n'.format(i+1, error_rate))
            print('Number of Epochs: {0}\tError Rate: {1}\n'.format(i+1, error_rate))
    
    #Testing
    xor_file.write("Testing the Perceptron\n")
    print("Testing the Perceptron\n")
    index = 0
    for arr in inputs:
        mlp.forward(arr, 'xor')
        xor_file.write('{0}Expected: {1}\tOutput: {2}\n'.format(arr, expected_output[index], mlp.O))
        print('{0}Expected: {1}\tOutput: {2}\n'.format(arr, expected_output[index], mlp.O))
        index +=1


# Main
# Were going to loop through different parameters to try identify patterns in the behaviour of the perceptron 
max_epochs = [5000, 10000]
learning_rate = [1, 0.75, 0.5]
xor_file = open('xor_results.txt', 'w')
for i in range(0,len(max_epochs)):
    xor_file.write('\n\nMax Epoch\n___________________________:{0} '.format(max_epochs[i]))
    for j in range(0,len(learning_rate)):
        XOR(max_epochs[i], learning_rate[j], xor_file)

# Closing the file
xor_file.close()