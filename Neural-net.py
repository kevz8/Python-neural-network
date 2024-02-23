import numpy as np

def sigmoid(i, deriv = False):
    if deriv:
        return i * (1 - i)
    return 1/(1 + np.exp(-i))

# Input
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

# Output
y = np.array([[1], [0], [0], [1], [0], [1], [1], [0]])

# Propogation

class Neural_net(object):
    def __init__(self):
        self.input_size = 3
        self.output_size = 1
        self.hidden_size = 3

        #weights
        self.W1 = 2 * np.random.random(size = (self.input_size, self.hidden_size)) - 1
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = 2 * np.random.random(size = (self.hidden_size, self.hidden_size)) - 1
        self.b2 = np.zeros((1, self.hidden_size))

        self.W3 = 2 * np.random.random(size = (self.hidden_size, self.output_size)) - 1
        self.b3 = np.zeros((1, self.output_size))

    def feedForward(self, X):
        #forward propagation
        z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(z1)

        z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(z2)

        z3 = np.dot(self.a2, self.W3) + self.b3
        output = sigmoid(z3)
        return output
    
    def backpropagation(self, X, y, r = 0.01):
        output = self.feedForward(X)
        a1 = self.a1
        a2 = self.a2
        error = y - output

        output_delta = error * sigmoid(output, deriv = True)
        self.W3 += np.dot(a2.T, output_delta) * r
        self.b3 += np.sum(output_delta, axis = 0, keepdims = True) * r

        hidden2_error = np.dot(output_delta, self.W3.T)
        hidden2_delta = hidden2_error * sigmoid(a2, deriv = True)
        self.W2 += np.dot(a1.T, hidden2_delta) * r
        self.b2 += np.sum(hidden2_delta, axis = 0, keepdims = True) * r

        hidden1_error = np.dot(hidden2_delta, self.W2.T)
        hidden1_delta = hidden1_error * sigmoid(a1, deriv = True)
        self.W1 += np.dot(X.T, hidden1_delta) * r
        self.b1 += np.sum(hidden1_delta, axis = 0, keepdims = True) * r


neural_network = Neural_net()

for j in range(200000):
    # Forward and backward pass
    neural_network.feedForward(X)
    neural_network.backpropagation(X, y)
print("W1:", neural_network.W1, "b1:", neural_network.b1)
print(" ")
print("W2:", neural_network.W2, "b2:", neural_network.b2)

# Print the final output after training
final_output = neural_network.feedForward(X)
print("Output after training:")
print(final_output)
