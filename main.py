import time

from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

from utils import nn

def main():

    # plt.figure(num=1,figsize=(5, 5))
    
    target = target_data(shape=32,coverage=75,plot_flag=False)
    
    # plt.savefig(f"target_data.png",)
    # return
    model = nn.Model()

    t1 = time.time()
    
    batch_size = 4
    
    shuffle(target)

    target_plot = target[0:int(len(target)/5)]

    for epoch in range(1000):
        
        plt.figure(num=2,figsize=(5, 5))
        show_prediction(shape=32,model=model,target=target_plot)

        # plt.savefig(f"{epoch}.png",)

        shuffle(target)

        inputx = []
        inputy = []
        counter = 0
        epoch_loss = 0
        for t in target:
            if counter == batch_size-1:
                running_loss = model.backprop_gradients(inputx,inputy)
                # running_loss = model.train(inputx,inputy)
                inputx = []
                inputy = []
                counter = 0
                epoch_loss+=running_loss
            inputx.append([t[0],t[1]])
            inputy.append(t[2])
            counter += 1
        print(epoch_loss)
            

        inputx = []
        inputy = []

    print(time.time()-t1)

def show_prediction(shape,model,scaling=1,target=[]):
    
    plt.clf()

    cmap = np.zeros((shape*scaling,shape*scaling))
    # xspace = [x/2 for x in range(shape*2) for y in range(shape*2)]
    # yspace = [y/2 for x in range(shape*2) for y in range(shape*2)]

    xspace = [x/shape for x in range(shape*scaling) for y in range(shape*scaling)]
    yspace = [y/shape for x in range(shape*scaling) for y in range(shape*scaling)]

    for x,y in zip(xspace,yspace):
        res = model.forward([x,y])
        # cmap[int(y*2),int(x*2)] = res[0]/res[1]
        if res[0]>res[1]:
            cmap[int(y*shape*scaling),int(x*shape*scaling)] = 1
        else:
            cmap[int(y*shape*scaling),int(x*shape*scaling)] = 0
    
    plt.imshow((cmap),cmap='jet', vmin=0, vmax=1,alpha=0.75)

    #also plot target
    for x,y,z in target:
        if z[0] == 1:
            plt.plot(x*shape,y*shape,'ro',markeredgecolor='k',markersize=4)
        else:
            plt.plot(x*shape,y*shape,'bo',markeredgecolor='k',markersize=4)

    plt.pause(0.01)

def target_data(shape:int, coverage:int, plot_flag:bool) -> list:

    #make target square sparse matrix with target values - [x,y,class]
    #x,y - positioin value
    #class = vector with 2 components where '1' indicates the class - [0,1] or [1,0]

    res = []
    for x in range(shape):
        for y in range(shape):
            if np.random.randint(1,101) < coverage:
                
                xx = np.array(x+np.random.randn()/5)/shape
                xx.astype(np.float64)

                yy = np.array(y+np.random.randn()/5)/shape
                yy.astype(np.float64)

                res.append([xx,yy])
    

    # # compute 'soft' sigmoidal boundary for class assignment, class is a 2 component vector
    for xy in res:
        if xy[1] < 0.25*np.sin(16*xy[0]) - xy[0] + 1:
            xy.append([1,0])
        else:
            xy.append([0,1])

    # circle
    # for xy in res:
    #     if np.sqrt((xy[0]-0.5)**2 + (xy[1]-0.5)**2) < 0.3:
    #         xy.append(np.array([1,0],dtype=np.float64))
    #     else:
    #         xy.append(np.array([0,1],dtype=np.float64))

    # just line
    # for xy in res:
    #     if xy[1] > xy[0]**3 + 0.1:
    #         xy.append(np.array([1,0],dtype=np.float64))
    #     else:
    #         xy.append(np.array([0,1],dtype=np.float64))

    if plot_flag:
        for x,y,z in res:
            if z[0] == 1:
                plt.plot(x,y,'ro')
            else:
                plt.plot(x,y,'bo')

        plt.show(block=False)
    plt.pause(1)
    return res

if __name__ == '__main__':
    main()
