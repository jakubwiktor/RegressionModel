import numpy as np
#working

class Model():
    
    #general model class which contains netork architecture and forwad and train calls given 'x' as datapoint. Train functionn will update weights of each layer.

    def __init__(self, num_layers = 3, num_features = [2,2,2,2]):

        #define netowork architecture and layers        
        
        self.Activation = Activation()
        self.Loss = Loss()

        self.L1 = Linear(features_in=num_features[0], features_out=num_features[1]) #first + hidden
        self.L2 = Linear(features_in=num_features[1], features_out=num_features[2]) #hidden
        self.L3 = Linear(features_in=num_features[2], features_out=num_features[3]) #output

        self.learning_rate = 0.003

    def forward(self,x):
        
        #forward pass on the data
        # out = self.L1.run(x)
        # out = self.Activation.RelU(out)
        # out = self.L2.run(x)
        # out = self.Activation.RelU(out)
        out = self.L3.run(x)
        # out = self.Activation.Sigmoid(out)

        return out

    def backprop_gradients(self,x,y):
        #calculate gradients with back pass.

        #for now harcode the derivatives of RelU and loss and network architecture
        pred = self.forward(x)

        # loss -> (y-x)^2, derivative -> 2(y-x)
        # sigmoid sig(x), derivative -> sig(x) * (1-sig(x))
        # RelU max(0,x), derivative -> 1 if x>0, else 0
        # weight w*input + b, derivative -> input
        # L3
        
        #get derivative of the cost
        cost_derivative = 2 * (pred-y)
        # cost_derivative = self.Activation.Sigmoid(cost_derivative)
        # cost_derivative = cost_derivative * (1-cost_derivative)

        #weight is [fatures_in, features_out]
        
        #L3
        L3_gradientsW = np.zeros((self.L3.features_in,self.L3.features_out))
        for i in range(len(self.L3.weights)):
            for j in range(len(self.L3.weights[i])): 
                L3_gradientsW[i][j] = cost_derivative[j] * x[i] 

        L3_gradientsB = np.zeros(self.L3.features_out)
        for i in range(len(self.L3.biases)):
                L3_gradientsB[i] = cost_derivative[i] * 1

        self.L3.weights -= L3_gradientsW * self.learning_rate
        self.L3.biases  -= L3_gradientsB * self.learning_rate

        # self.L2.weights -= L2_gradientsW * self.learning_rate
        # self.L2.biases  -= L2_gradientsB * self.learning_rate

        # for i in range(len(self.L3.weights)):
        #     for j in range(len(self.L3.weights[i])): 
        #         L3_gradients[i][j] = cost_derivative[j] * self.L3.node_values[j]
 

        return np.sum(np.power(pred-y,2))
    
    def train(self,x,y):
        #this function will take data and work on the gradients of the layers. Gradients are stored in the Linear layer class.
        
        #simplest thing is to loop each Linear layer and run a datapoint for each weight in self.L1.weights and then add small
        #value and run the same point again. This will compute a gradient. 

        #x i input vector with position of datapoint data, y is a class vector
        
        h = 0.000001
        current_loss = self.Loss.loss(self.forward(x), y)

        #get gradients for weights and biases
        # for i in range(len(self.L1.weights)):
        #     for j in range(len(self.L1.weights[i])):
        #         self.L1.weights[i][j] += h
        #         new_loss = 0
        #         for one_x, one_y in zip(x,y):
        #             new_loss += self.Loss.loss(self.forward(one_x), one_y)
        #         # new_loss = new_loss/len(x)
        #         self.L1.weights[i][j] -= h
        #         self.L1.weights_gradients[i][j] = (new_loss - current_loss)/h
                
        for i in range(len(self.L2.weights)):
            for j in range(len(self.L2.weights[i])):
                self.L2.weights[i][j] += h
                new_loss = self.Loss.loss(self.forward(x), y)
                self.L2.weights_gradients[i][j] = (new_loss - current_loss)/h
                self.L2.weights[i][j] -= h
    
        for i in range(len(self.L3.weights)):
            for j in range(len(self.L3.weights[i])):
                self.L3.weights[i][j] += h
                new_loss = self.Loss.loss(self.forward(x), y)
                self.L3.weights_gradients[i][j] = (new_loss - current_loss)/h
                self.L3.weights[i][j] -= h
                            
        # for i in range(len(self.L1.biases)):
        #     self.L1.biases[i] += h
        #     new_loss = 0
        #     for one_x, one_y in zip(x,y):
        #         new_loss += self.Loss.loss(self.forward(one_x), one_y)
        #     self.L1.biases_gradients[i] = (new_loss - current_loss)/h
        #     self.L1.biases[i] -= h
                
        for i in range(len(self.L2.biases)):
            self.L2.biases[i] += h
            new_loss = self.Loss.loss(self.forward(x), y)
            self.L2.biases_gradients[i] = (new_loss - current_loss)/h
            self.L2.biases[i] -= h

        for i in range(len(self.L3.biases)):
            self.L3.biases[i] += h
            new_loss = self.Loss.loss(self.forward(x), y)
            self.L3.biases_gradients[i] = (new_loss - current_loss)/h
            self.L3.biases[i] -= h

        # self.L1.weights -= self.L1.weights_gradients*self.learning_rate        
        self.L2.weights -= self.L2.weights_gradients*self.learning_rate
        self.L3.weights -= self.L3.weights_gradients*self.learning_rate
        
        # self.L1.biases  -= self.L1.biases_gradients*self.learning_rate
        self.L2.biases  -= self.L2.biases_gradients*self.learning_rate
        self.L3.biases  -= self.L3.biases_gradients*self.learning_rate

        return current_loss

#####
# other casses
#####      


class Linear():

    #linear layer for neural network project

    def __init__(self, features_in:int, features_out:int):
        
        #preallocate random vectors for the weights and the biases
        self.features_in = features_in
        self.features_out = features_out
        
        self.weights = np.random.randn(self.features_in,self.features_out)#as a np matrix 1st - features_in, 2nd - features_out
        self.weights = self.weights / (np.sqrt(features_in)*100)
        self.weights.astype(np.float64)

        self.biases = np.zeros(self.features_out)
        self.biases.astype(np.float64)

        self.weights_gradients = np.zeros((self.features_in,self.features_out)) #as a np matrix 1st - features_in, 2nd - features_out
        self.weights_gradients.astype(np.float64)
        
        self.biases_gradients = np.zeros(self.features_out) #as a np matrix 1st - features_in, 2nd - features_out
        self.biases_gradients.astype(np.float64)

        self.node_values = np.zeros(self.features_out) #to store for gradient descent

    def run(self, x): 
    #for each feature in 'x' multiply by relevant weights and add up, then and bias

        #numpy vectorization        
        # output  = np.dot(self.weights.T, x) + self.biases
        output = np.zeros(self.features_out)
        for outgoing in range(self.features_out):
            for arriving, _ in enumerate(x):
                output[outgoing] += x[arriving] * self.weights[arriving, outgoing]    
            output[outgoing] += self.biases[outgoing]      
        
        self.node_values = output
        
        return output

class Activation():
    
    #RelU activation layer

    def __init__(self):
        pass

    def RelU(self,x): 
        return [max(0,r) for r in x]
    
    def Sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def TanH(self,x):
        res=[]
        for r in x:
            res.append((np.exp(r) - np.exp(-r)) / (np.exp(r) + np.exp(-r)))
        return res

class Loss():
    #define loss function for neural netowrk
    
    def __init__(self):
        pass

    def loss(self,x,y):
        #loss as a square of difference between the output of the net (X) and the target data (Y). 
        #in this case the output is 0/1 score. 'X'
        this_loss = np.sum(np.power(y-x,2))
        # print(x,y,this_loss)
        return this_loss

