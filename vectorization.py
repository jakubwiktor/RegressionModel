import numpy as np
import time

def main():
    #try to vectorize weight calcilation in numpy
    num_in = 128
    num_out = 8
    input_features = np.ones(num_in).astype(np.float64)
    weights = np.random.randn(num_in,num_out).astype(np.float64)#as a np matrix 1st - features_in, 2nd - features_out
    biases = np.random.randn(num_out).astype(np.float64)
    
    #linear programming
    t1 = time.time()
    for x in range(1000):
        res = []
        for i in range(num_out):
            res_tmp = biases[i]
            for j in range(num_in):
                res_tmp += input_features[j] * weights[j,i]
            res.append(res_tmp)
    print(res)
    print(time.time()-t1)

    t2 = time.time()
    for x in range(1000):
        this=np.dot(input_features,weights) + biases #or matmul
    print(this)
    print(time.time()-t2)

    print(this==res)

if __name__ == "__main__":
    main()