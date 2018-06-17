import numpy as np
from multiprocessing import Pool 
import time

# sigmoid function
def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)
	return 1/(1+np.exp(-x/10))
	
    
hid1 = 4  #neurons on the first hidden layer
hid2 = 4  #neurons on the second hidden layer
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

#initialize fisrt hidden layer
FH = np.zeros((comb,hid1))
FH2 = np.zeros((comb,hid1))

#initialize second hidden layer
SH = np.zeros((comb,hid2))
SH2 = np.zeros((comb,hid2))

#initialize out layer
O = np.zeros((comb,1))
O2 = np.zeros((comb,1))


# randomly initialize weights
w0 = r*(np.random.random((init,hid1)) - 0.5)
wf1 = r*(np.random.random((hid1,hid2)) -0.5)
ws1 = r*(np.random.random((hid2,1)) - 0.5)

#time measurement
start_time = time.time()

#initialize number of threads
p=Pool(1)


for j in xrange(5000):

    I = X
#find hidden layer
    FH1 = np.dot(I,w0)
#parallel function
    FH2 = p.map(nonlin,FH1)

#change type
    for i in range (comb):
        FH[i][:] = FH2[i][:]

#find hidden layer
    SH1 = np.dot(FH,wf1)
#parallel function
    SH2 = p.map(nonlin,SH1)

#change type
    for i in range (comb):
        SH[i][:] = SH2[i][:]

#find out layer
    O1 = np.dot(SH,ws1)
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

#find  second layer error
    SH_error = O_delta.dot(ws1.T)
    
#find hidden layer delta
    SH_delta = SH_error * nonlin(SH,deriv=True)

#find first layer error
    FH_error = SH_delta.dot(wf1.T)
    
#find hidden layer delta
    FH_delta = FH_error * nonlin(FH,deriv=True)

#find new weights
    ws1 += SH.T.dot(O_delta)
    wf1 += FH.T.dot(SH_delta)
    w0 += I.T.dot(FH_delta)
   
#cycle interruption
    if np.mean(np.abs(O_error)) < 0.1:
         break
    print "out"
    print j
print O
print "time"
print (time.time()-start_time)    

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
FH =nonlin(np.dot(I,w0))
SH = nonlin(np.dot(FH,wf1)) 
O = nonlin(np.dot(SH,ws1))
print O

