import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

def reLu(Z):
   return np.maximum(0, Z)

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def reLu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid_derivative(s):
    return s * (1 - s)

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    loss = np.sum(np.abs(pred-real)) / 1168
    return loss



class NN:

    def __init__(self, X, y):
        super().__init__()
        self.X = X.T
        #print(X.shape)
        self.y = y
        #print(y)
        neurons = 15
        self.learning_rate = 0.5

        num_features = X.shape[1]
        num_outputs = y.shape[1]
        #print(num_outputs)

        self.W1 = np.random.randn(neurons, num_features) *0.01
        #print(self.W1.shape)
        self.b1 = np.random.randn(neurons, 1)
        #print(self.b1.shape)
        #print(self.bias1)

        self.W2 = np.random.randn(1, neurons) * 0.01
        self.b2 = np.random.randn(1, 1)
        #print(self.b2)


    
    def forward(self):
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        #print(Z1)
        self.A1 = reLu(self.Z1)
        #print(self.A1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        #print(Z2)
        self.A2 = sigmoid(self.Z2)
        #print(self.A2)

    def back_propagate(self):
        #print(self.y)
        #print(self.A2)
        loss = error(self.A2, self.y)
        print('Error :', loss)

        delta_Z2 = self.A2 - self.y
        delta_W2 = np.dot(delta_Z2, self.A1.T) / self.y.shape[1]
        delta_b2 = np.sum(delta_Z2, axis = 1, keepdims=True)

        delta_Z1 = np.dot(self.W2.T, delta_Z2) * reLu_derivative(self.Z1)
        delta_W1 = np.dot(delta_Z1, self.X.T) / self.y.shape[1]
        delta_b1 = np.sum(delta_Z1, axis = 1, keepdims=True)

        self.W1 -= self.learning_rate * delta_W1
        self.W2 -= self.learning_rate * delta_W2
        self.b1 -= self.learning_rate * delta_b1
        self.b2 -= self.learning_rate * delta_b2

df = pd.read_csv('../data/housepricedata.csv', delimiter = ',')
#data.head()
#print(data.head())

data = df.to_numpy()
#print(data)

features_train = data[:1168, :-1]
features_test = data[1168:, :-1]
labels_train = data[:1168, -1]
labels_test = data[1168:, -1]

labels_train = labels_train.reshape((1, labels_train.shape[0]))
labels_test = labels_test.reshape((1, labels_test.shape[0]))

model = NN(features_train, labels_train)

max_iter = 1000
for i in range(max_iter) :
    model.forward()
    model.back_propagate()
