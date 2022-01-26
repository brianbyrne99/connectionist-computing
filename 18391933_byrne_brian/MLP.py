import numpy as np


class MLP:

    def __init__(self, NI, NH, NO):
        self.NI = NI  # number of inputs
        self.NH = NH  # number of hidden units
        self.NO = NO  # number of outputs
        self.w1 = np.array  # array containing the weights in the lower layer
        self.w2 = np.array  # array containing the weights in the upper layer
        self.dw1 = np.array  # arrays containing the weight *changes* to be applied onto W1 and W2
        self.dw2 = np.array
        self.z1 = np.array  # arrays containing the activations for the lower layer
        self.z2 = np.array  # upper layer
        # array where the values of the hidden neurons are stored – need these saved to compute dW2
        self.H = np.array
        self.O = np.array  # array where the outputs are stored

    def randomise(self):
        # Initialises W1 and W2 to small random values
        self.w1 = np.random.uniform(0, 1, (self.NI, self.NH))
        self.w2 = np.random.uniform(0, 1, (self.NH, self.NO))

        # set dW1 and dW2 to all zeroes
        self.dw1 = np.random.uniform(0, 0, (self.NI, self.NH))
        self.dw2 = np.random.uniform(0, 0, (self.NH, self.NO))

    def activateSigmoid(self, arr):
        return 1 / (1 + np.exp(-arr))

    def activateTanh(self, arr):
        return (2 / (1 + np.exp(arr * -2))) - 1

    def activateSigmoidDerivative(self, arr):
        return np.exp(-arr) / (1 + np.exp(-arr)) ** 2

    def activateTanhDerivative(self, arr):
        return 1 - np.tanh(arr)**2

    # Here we will proviced the option of using (sigmoid [0,1] range) or tanh ([-1,1]) as activation functions
    def forward(self, input_vectors, problem_type):
        # Input vector I is processed to produce an output, which is stored in O[]
        self.z1 = np.dot(input_vectors, self.w1)
        # We will use the sigmoidal function for xor as out put [0,1]
        if problem_type == 'xor':
            self.H = self.activateSigmoid(self.z1)
            self.z2 = np.dot(self.H, self.w2)
            self.O = self.activateSigmoid(self.z2)
        # We will use the tanh function for xor as out put [-1,1]
        if problem_type == 'sin':
            self.H = self.activateTanh(self.z1)
            self.z2 = np.dot(self.H, self.w2)
            self.O = self.activateTanh(self.z2)

    # this function learns which weights are better and how much prediciton has deviated from the actual output
    def backwards(self, input_vectors, target, problem_type):
        error_in_output = np.subtract(target, self.O)

        if problem_type == 'xor':
            O_activation = self.activateSigmoidDerivative(self.z2)
            H_activation = self.activateSigmoidDerivative(self.z1)

        if problem_type == 'sin':
            O_activation = self.activateTanhDerivative(self.z2)
            H_activation = self.activateTanhDerivative(self.z1)

        dw2_O = np.multiply(error_in_output, O_activation)
        self.dw2 = np.dot(self.H.T, dw2_O)

        dw1_H = np.multiply(np.dot(dw2_O, self.w2.T), H_activation)
        self.dw1 = np.dot(input_vectors.T, dw1_H)
        return np.mean(np.abs(error_in_output))

# simply updates the weights to edge towards an optimal solution.
    def updateWeights(self, learning_rate):
        self.w1 = np.add(self.w1, learning_rate * self.dw1)
        self.w2 = np.add(self.w2, learning_rate * self.dw2)
        self.dw1 = np.array
        self.dw2 = np.array

    def printDetails(self):
        instance_variables = vars(self)
        print(instance_variables)
