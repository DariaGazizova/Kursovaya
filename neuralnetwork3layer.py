
'''
http://iamtrask.github.io/2015/07/12/basic-python-network/
was used for development of this code
'''


import numpy as np
from multiprocessing import Pool 
import time

# sigmoid function
def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)
	return 1/(1+np.exp(-x/10))
	
    
hid = 4  #neurons on the hidden layer
comb = 6 #number of combinations
init = 3 #neurons on the initial layer 
r = 2

f = open('init.txt','r')
lineinit = f.readlines()
#initialize X from file
X = np.zeros((comb,init))

for i in range(0,comb):
    lineinit1 = lineinit[i].split()
    lineinit2 = lineinit1[20-init:20]
    lineinit3 = [int(x) for x in lineinit2]
    X[i][:] = lineinit3 

f.close()

f = open('result.txt','r')
lineresult = f.readlines()
#initialize y from file
y = np.zeros((comb,1))
for i in range(0,comb):
    lineresult1 = lineresult[i].split()
    lineresult2 = lineresult1[0:1]
    lineresult3 = [int(x) for x in lineresult2]
    y[i] = lineresult3

f.close()

#initialize hidden layer
H = np.zeros((comb,hid))

H2  =np.zeros((comb,hid))

#initialize out layer
O = np.zeros((comb,1))
O2 = np.zeros((comb,1))

# randomly initialize weights
w0 = r*(np.random.random((init,hid)) - 0.5)
w1 = r*(np.random.random((hid,1)) - 0.5)

#time measurement
start_time = time.time()

#initialize number of threads
p = Pool(4)

for j in xrange(5000):

    I = X
#find hidden layer
    H1 = np.dot(I,w0)
#parallel function
    H2 = p.map(nonlin,H1)

#change type   
    for i in range (comb):
        H[i][:] = H2[i][:]

#find out layer
    O1 = np.dot(H,w1)
#parallel function
    O2 = p.map(nonlin,O1)

#change type
    for i in range (comb):
        O[i][:] = O2[i][:]

#find result error
    O_error = y - O
    
    print "Error:" + str(np.mean(np.abs(O_error)))
        
#find results delta
    O_delta = O_error*nonlin(O,deriv=True)

# how much did each H value contribute to the O error (according to the weights)?
    H_error = O_delta.dot(w1.T)
    
#find hidden layer delta
    H_delta = H_error * nonlin(H,deriv=True)

#find new weights
    w1 += H.T.dot(O_delta)
    w0 += I.T.dot(H_delta)


#cycle interruption
    if np.mean(np.abs(O_error)) < 0.1:
        break

    print j
print O

print (time.time()-start_time)    

f= open('weights.txt', 'w')
f.write(str(init))
f.write('\n')
f.write(str(hid))
f.write('\n')
for i in range(init):
    for j in range(hid): 
        f.write(str(w0[i][j]))
        f.write(" ")
    f.write('\n')
for i in range(hid):
    #for 
    f.write(str(w1[i][0]))
    f.write('\n')
f.close()

f = open('init.txt','r')
line1 = f.readlines()
#calculation test combination
test = np.zeros((1,init))
W = line1[init+1].split()
W2 = W[20-init:20]
W3 = [int(x) for x in W2]
test = W3
print test
I=test
H =nonlin(np.dot(I,w0)) 
O = nonlin(np.dot(H,w1))
print O

