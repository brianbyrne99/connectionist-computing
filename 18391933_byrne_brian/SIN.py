import numpy as np
from MLP import MLP

def SIN(max_epochs, learning_rate, sin_file):
    # intialising MultiLayer perceptron for Sin function
    num_inputs = 4
    num_hidden_units = 50
    num_outputs =1
    mlp = MLP(num_inputs, num_hidden_units, num_outputs)

    # initialising our input arrays and calculating the expected outputs
    inputs = []
    expected_outputs = []

    for i in range(0, 500):
        vector = list(np.random.uniform(-1.0, 1.0, 4))
        inputs.append(vector)

    inputs=np.array(inputs)
    for i in range(0, 500):
        expected_outputs.append(np.sin([inputs[i][0] - inputs[i][1] + inputs[i][2] - inputs[i][3]]))

    mlp.randomise()

    #Writing results to a log file so we can perform a deeper statistical analysis later on
    sin_file.write('MLP structure for SIN : #inputs:{0}   #Hidden Units:{1}   #Outputs:{2}\n'.format(num_inputs, num_hidden_units, num_outputs))
    sin_file.write('Maximum number of Epochs:{0}\n'.format(max_epochs))
    sin_file.write('Learning rate:{0}\n'.format(learning_rate))

    print('MLP structure for SIN : #inputs:{0}   #Hidden Units:{1}   #Outputs:{2}\n'.format(num_inputs, num_hidden_units, num_outputs))
    print('Maximum number of Epochs:{0}\n'.format(max_epochs))
    print('Learning rate:{0}\n'.format(learning_rate))

    # Testing before training
    sin_file.write("Before Training Analysis")
    print("Before Training Analysis")
    # As this is a much larger dataset than that of the last question, we will use a small sample of 
    # of the data set to give an idea of pre training performance
    for i in range(0,20):      
        mlp.forward(inputs[i], 'sin')
        sin_file.write('{0}Expected: {1}\tOutput: {2}\n'.format(inputs[i], expected_outputs[i], mlp.O))
        print('{0}Expected: {1}\tOutput: {2}\n'.format(inputs[i], expected_outputs[i], mlp.O))


    #Training
    sin_file.write("Training the Perceptron\n")
    print("Training the Perceptron\n")
    for i in range(0, max_epochs):
        mlp.forward(inputs[:400], 'sin')
        error_rate = 0
         # We're going to use an 80/20 train test split for this neural network
        error_rate = mlp.backwards(inputs[:400], expected_outputs[:400], 'sin')
        # update the weights with learning weight
        mlp.updateWeights(learning_rate)

        if (i + 1) % (max_epochs / 20) == 0:
            sin_file.write('Number of Epochs: {0}\tError Rate: {1}\n'.format(i+1, error_rate))
            print('Number of Epochs: {0}\tError Rate: {1}\n'.format(i+1, error_rate))


    #Testing
    # As this is such a large train and test set we will calcualte the mean accuracy and differences
    # between predicted output and expected output.
    sin_file.write("Testing the Perceptron\n")
    print("Testing the Perceptron\n")
    diff = 0
    for i in range(400, len(inputs)):
        mlp.forward(inputs[i], 'sin')
        #Aggregation of differences between target and prediciton.
        diff = diff + np.abs(expected_outputs[i][0] - mlp.O[0])

    mean_accuracy = round(1-(diff/100), 4) # is the population of the test set.
    sin_file.write('Total aggregated diffence between the test predictions and actual output : {0}'.format(diff))
    print('Total aggregated diffence between the test predictions and actual output : {0}\n'.format(diff))
    sin_file.write('\nOn average, the neural net had an accuracy of {0}'.format(mean_accuracy))
    print('On average, the neural net had an accuracy of {0}'.format(mean_accuracy))

# Main
sin_file = open('sin_results.txt', 'w')
for i in [0.1,0.01,0.0001]:
    sin_file.write('\n : {0}'.format(i))
    SIN(30000, i, sin_file)
sin_file.close()