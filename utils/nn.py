import numpy as np
#working

class Model():
    
    #general model class which contains netork architecture and forwad and train calls given 'x' as datapoint. Train functionn will update weights of each layer.

    def __init__(self, num_layers = 3, num_features = [2,8,8,2]):

        #define netowork architecture and layers        
        
        self.Activation = Activation()
        self.Loss = SquareLoss()

        self.features_0 = np.zeros(num_features[0])
        self.features_1 = np.zeros(num_features[1])
        self.features_2 = np.zeros(num_features[2])
        self.features_3 = np.zeros(num_features[3])

        self.L1 = Linear(features_in=self.features_0, features_out=self.features_1) #first + hidden
        self.L2 = Linear(features_in=self.features_1, features_out=self.features_2) #hidden
        self.L3 = Linear(features_in=self.features_2, features_out=self.features_3) #output


    def forward(self,x):
        
        #forward pass on the data
        out = self.L1.run(x)
        out = self.Activation.RelU(out)
        out = self.L2.run(out)
        out = self.Activation.RelU(out)
        out = self.L3.run(out)
        out = self.Activation.RelU(out)
        
        # print(out)
        # max_out = sum(out)
        # out = [x/max_out for x in out]
        return out

    def train(self,x,y):
        #this function will take data and work on the gradients of the layers. Gradients are stored in the Linear layer class.
        
        #simplest thing is to loop each Linear layer and run a datapoint for each weight in self.L1.weights and then add small
        #value and run the same point again. This will compute a gradient. 

        #x i input vector with position of datapoint data, y is a class vector
        
        h = 0.00001 #learning rate hardcoded for now
        learning_rate = 0.001
        current_loss = self.Loss.loss(self.forward(x), y)

        #get gradients for weights and biases
        for i in range(len(self.L1.weights)):
            for j in range(len(self.L1.weights[i])):
                self.L1.weights[i][j] += h
                new_loss = self.Loss.loss(self.forward(x), y)
                gradient = (new_loss - current_loss)/h
                # print(gradient)
                self.L1.weights_gradients[i][j] = gradient*learning_rate
                self.L1.weights[i][j] -= h

        for i in range(len(self.L2.weights)):
            for j in range(len(self.L2.weights[i])):
                self.L2.weights[i][j] += h
                new_loss = self.Loss.loss(self.forward(x), y)
                gradient = (new_loss - current_loss)/h
                self.L2.weights_gradients[i][j] = gradient*learning_rate
                self.L2.weights[i][j] -= h
    
        for i in range(len(self.L3.weights)):
            for j in range(len(self.L3.weights[i])):
                self.L3.weights[i][j] += h
                new_loss = self.Loss.loss(self.forward(x), y)
                gradient = (new_loss - current_loss)/h
                self.L3.weights_gradients[i][j] = gradient*learning_rate
                self.L3.weights[i][j] -= h
                
        for i in range(len(self.L1.biases)):
            self.L1.biases[i] += h
            new_loss = self.Loss.loss(self.forward(x), y)
            gradient = (new_loss - current_loss)/h
            self.L1.biases_gradients[i] = gradient*learning_rate
            self.L1.biases[i] -= h

        for i in range(len(self.L2.biases)):
            self.L2.biases[i] += h
            new_loss = self.Loss.loss(self.forward(x), y)
            gradient = (new_loss - current_loss)/h
            self.L2.biases_gradients[i] = gradient*learning_rate
            self.L2.biases[i] -= h

    
        for i in range(len(self.L3.biases)):
            self.L3.biases[i] += h
            new_loss = self.Loss.loss(self.forward(x), y)
            gradient = (new_loss - current_loss)/h
            self.L3.biases_gradients[i] = gradient*learning_rate
            self.L3.biases[i] -= h

        self.L1.weights -= self.L1.weights_gradients
        self.L2.weights -= self.L2.weights_gradients
        self.L3.weights -= self.L3.weights_gradients

        self.L1.biases  -= self.L1.biases_gradients
        self.L2.biases  -= self.L2.biases_gradients
        self.L3.biases  -= self.L3.biases_gradients

        return None

#####
# other casses
#####      


class Linear():

    #linear layer for neural network project

    def __init__(self, features_in:int, features_out:int):
        
        #preallocate random vectors for the weights and the biases
        self.features_in = features_in
        self.features_out = features_out
        
        self.weights = np.random.rand(self.features_in.size, self.features_out.size)/100 #as a np matrix 1st - features_in, 2nd - features_out
        self.biases = np.random.rand(self.features_out.size)/100
        
        self.weights_gradients = np.zeros((self.features_in.size, self.features_out.size)) #as a np matrix 1st - features_in, 2nd - features_out
        self.biases_gradients = np.zeros(self.features_out.size) #as a np matrix 1st - features_in, 2nd - features_out

    def run(self, x): 
    #for each feature in 'x' multiply by relevant weights and add up, then and bias
        
        output = np.zeros(self.features_out.size)
        
        for outgoing, _ in enumerate(self.features_out):
            weighted_feature = 0            
            for arriving, _ in enumerate(x):
                weighted_feature += x[arriving] * self.weights[arriving,outgoing]
            weighted_feature += self.biases[outgoing]
            output[outgoing] = weighted_feature
        
        return output


class Activation():
    
    #RelU activation layer

    def __init__(self):
        pass

    def RelU(self,x:float): 
        return [max(0,n) for n in x]


class SquareLoss():
    #define loss function for neural netowrk
    
    def __init__(self):
        pass

    def loss(self,x:float,y:float):
        #loss as a square of difference between the output of the net (X) and the target data (Y). 
        #in this case the output is 0/1 score. 'X'

        this_loss = np.sum([(xx-yy)**2 for xx,yy in zip(x,y)])
        # print(x,y,this_loss)
        return this_loss

