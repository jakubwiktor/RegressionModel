import time

from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

from utils import nn

def main():

    plt.figure(1)
    target = target_data(shape=32,coverage=40,plot_flag=True)

    
    model = nn.Model()
    loss = nn.SquareLoss()

    t1 = time.time()
    for epoch in range(1,100):
        print(epoch)
        plt.figure(2)
        show_prediction(shape=32,model=model)
        
        shuffle(target)

        for t_counter, t in enumerate(target):
            model.train([t[0],t[1]],t[2])
            
            if t_counter %50:
                print( loss.loss( model.forward([t[0],t[1]]), t[2]) ) 
        

    print(time.time()-t1)

def show_prediction(shape, model):
    
    # fig = plt.figure()
    
    xspace = [x for x in range(shape) for y in range(shape)]
    yspace = [y for x in range(shape) for y in range(shape)]
    counter = 0
    for x,y in zip(xspace,yspace):
        a,b = model.forward([x,y])
        if a>b:
            color = 'xr'
        else:
            color = 'xb'
        
        plt.plot(x,y,color)
        
        counter += 1
    plt.show(block=False)
    plt.pause(1)

def target_data(shape:int, coverage:int, plot_flag:bool) -> list:

    #make target square sparse matrix with target values - [x,y,class]
    #x,y - positioin value
    #class = vector with 2 components where '1' indicates the class - [0,1] or [1,0]

    res = []
    for x in range(shape):
        for y in range(shape):
            if np.random.randint(1,101) < coverage:
                tmp = [x+np.random.randn()/5,y+np.random.randn()/5]
                res.append(tmp)

    #compute 'soft' sigmoidal boundary for class assignment, class is a 2 component vector
    for xy in res:
        if xy[1] < 5*np.sin(0.5*xy[0]) - 0.25*xy[0] + shape/2:
            xy.append([1,0])
        else:
            xy.append([0,1])

    if plot_flag:
        for x,y,z in res:
            if z[0] == 1:
                plt.plot(x,y,'ro')
            else:
                plt.plot(x,y,'go')

        plt.show(block=False)

    return res

if __name__ == '__main__':
    main()
