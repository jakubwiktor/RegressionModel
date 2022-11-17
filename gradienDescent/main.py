import time
import random
import math
import copy

import matplotlib.pyplot as plt

def main():

    plt.ion()
    figure, ax = plt.subplots(figsize=(5, 5))

    x = fractional_space(x1 = -2.7, x2 = 2.2, step = 0.1)
    y = [function(n) for n in x]

    line1, = ax.plot(x, y)
    point1, = ax.plot(0,function(0),'o')
    line_diff, = ax.plot(0,0)

    figure.canvas.draw()

    current_x = random.choice(x)
    current_x = x[0]
    learning_rate = 0.0001

    while True:

        #update the plot of current 'x'
        point1.set_xdata(current_x)
        point1.set_ydata(function(current_x))
        
        #compute derivative of the function
        derivative = compute_derivative(current_x)

        len_factor = 0.5 / (1+derivative**2)**(1/2)

        y0 = function(current_x) - derivative*len_factor
        y1 = function(current_x) + derivative*len_factor

        #update derivative
        line_diff.set_xdata([current_x-1*len_factor, current_x+1*len_factor])
        line_diff.set_ydata([y0, y1])

        #plot
        figure.canvas.draw()
        figure.canvas.flush_events()

        current_x -= derivative*learning_rate
        #pause for iterative plotting
        plt.pause(0.01)
        
        if abs(derivative) < 0.001:
            break

def compute_derivative(x:float) -> float:
    #compute derivative of a function
    h = 0.00001
    y = function(x)
    diff = function(x+h)
    return (diff-y)/h

def function(x:float) -> float:
    #simple function
    return (0.2)*(x**4) + (0.1)*(x**3) - x**2 + 2

def function_derivative(x:float) -> float:
    #derivative of 'simple function'
    return (0.8)*(x**3) + (0.3)*(x**2) - 2*x 

def fractional_space(x1:float,x2:float,step:float) -> list():
    #make linspace with fractional step from x1 to x2
    assert x1 < x2, \
        'x1 should be larger than x2'
    
    out = []
    while x1 < x2+step:
        out.append(x1)
        x1 += step
    return out
    
if __name__ == '__main__':
    main()