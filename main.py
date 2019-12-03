import random
import copy
import math

inner_nodes = 30
output_nodes = 7
signal_const = 1
bias1 = 0.5
bias2 = 0.75
alpha = 0.2
trainHard = 100


##sigmoids 
def bipolarSigmoid(x):
    return (2 / (1 + math.exp(-1*x ))) - 1

def bipolarSigmoidx(x):
    return 0.5*((1+bipolarSigmoid(x))*(1-bipolarSigmoid(x)) )




##node object for net
class Node():
    def __init__ (self):
        self.prevNode = []
        self.nextNode = []
        self.signal = 0
        self.weight = []
        self.error = 0
        self.deltaWeight = []
        self.y_in = 0
        self.bias1 = bias1
        self.bias2 = bias2
        self.deltabias1 = 0
        self.deltabias2 = 0

    def setSignal(self,sig):
        self.signal = sig

    def setWeight(self,wt):
        self.weight = wt

    def setPrevNode(self,prev):
        self.prevNode.append(prev)

    def setNextNode(self,nex):
        self.nextNode.append(nex)

class Net():
    def __init__(self):
        self.net = []
    def buildNet(self):
        ##output layer
        outputNodes = []
        for i in range(output_nodes):
            node = Node()
            outputNodes.append(node)
    
        ##median layer
        medianNodes = []
        for i in range(inner_nodes):
            node = Node()
            for j in outputNodes:
                node.weight.append(random.uniform(-1,1))
            medianNodes.append(node)


        ##input layer
        inputNodes = []
        for i in range(63):
            node = Node()
            for j in medianNodes:
                node.weight.append(random.uniform(-1,1))
            inputNodes.append(node)
        
        self.net.append(inputNodes)
        self.net.append(medianNodes)
        self.net.append(outputNodes)
    
    def feed(self, chars):
        count = 0
        for i in chars:
            if i != '.':
                self.net[0][count].signal = 1 * signal_const
            else:
                self.net[0][count].signal = -1 * signal_const
            count = count+1

        self.feedForward()

    def feedForward(self):
        ##first layer
        count = 0
        for i in self.net[1]:
            sum = 0
            for j in self.net[0]:
                sum = sum + (j.signal*j.weight[count])

            i.y_in = i.bias1 + sum
            i.signal = bipolarSigmoid(sum)
            count = count + 1

        ##second layer
        count = 0
        for i in self.net[2]:
            sum = 0
            for j in self.net[1]:
                sum = sum + (j.signal*j.weight[count])
            i.y_in = i.bias2 + sum
            i.signal = bipolarSigmoid(i.y_in)
            count = count+1
    
    def backProp(self, vector):
        count = 0
        for i in self.net[2]:
            i.error = (vector[count] - i.signal)*bipolarSigmoidx(i.y_in)
            for j in self.net[1]:
                j.deltaWeight.append(alpha * i.error * j.signal)
            i.deltabias2 = alpha * i.error

            count = count+1

        for i in self.net[1]:
            sum = 0
            count = 0
            for j in self.net[2]:
                sum = sum + j.error*i.weight[count]
            i.error = sum * bipolarSigmoidx(i.y_in)
            for j in self.net[0]:
                j.deltaWeight.append(alpha * i.error * j.signal)
            i.deltabias1 = alpha * i.error
            count = count+1


        self.updateNet()
    
    def updateNet(self):
        for i in self.net[0]:
            count = 0
            for j in i.weight:
                i.weight[count] = j + (i.deltaWeight.pop(0))
                count = count+1
            
        for i in self.net[1]:
            count = 0
            for j in i.weight:
                i.weight[count] = j + (i.deltaWeight.pop(0))
                count = count+1
            i.bias1 = i.bias1 + i.deltabias1

        for i in self.net[2]:
            i.bias2 = i.bias2 + i.deltabias2
    

                



            
            



netw = Net()
netw.buildNet()
f = open('HW3_Training.txt', 'r')
##train
for q in range(trainHard):
    target = [[1,-1,-1,-1,-1, -1, -1],[-1,1,-1,-1,-1, -1, -1],[-1,-1, 1,-1,-1, -1, -1],[-1,-1,-1, 1,-1, -1, -1],[-1,-1,-1,-1, 1, -1, -1],[-1,-1,-1,-1,-1, 1, -1],[-1,-1,-1,-1,-1, -1, 1]] 
    ##process
    letter = 0
    count= 0 
    chars = []
    for line in f:
        line = line.rstrip()
        if count < 8:
            for i in line:
                chars.append(i)
            count = count +1
        else:
            ##feed
            netw.feed(chars)
            if letter < 7:
                netw.backProp(target[letter])
                letter = letter + 1
            else:
                letter = 0
                netw.backProp(target[letter])
                letter = letter + 1
            ##continue
            chars = []
            count = 0
    
    f.seek(0)
    q = q+1

f.close()

##test
f = open('HW3_Testing.txt', 'r')
target = [[1,-1,-1,-1,-1, -1, -1],[-1,1,-1,-1,-1, -1, -1],[-1,-1, 1,-1,-1, -1, -1],[-1,-1,-1, 1,-1, -1, -1],[-1,-1,-1,-1, 1, -1, -1],[-1,-1,-1,-1,-1, 1, -1],[-1,-1,-1,-1,-1, -1, 1]] 
##process
letter = 0
count= 0 
chars = []
numRight = 0
total = 0
for line in f:
    line = line.rstrip()
    print(line)
    if count < 8:
        for i in line:
            chars.append(i)
        count = count +1
    else:
        ##feed
        netw.feed(chars)
        temp = copy.deepcopy(netw.net[2])
        print('---------------------------')
        print("for letter " + str(letter)+ ":")
        final = []
        count  = 0
        for i in temp:
            final.append([i.signal,count])
            count = count+1
        temp = sorted(final).pop()
        print(temp)
        asc = temp[1]
        if asc == 0:
            asc = 'A'
        elif asc == 1:
            asc = 'B'
        elif asc == 2:
            asc = 'C'
        elif asc == 3:
            asc = 'D'
        elif asc == 4:
            asc = 'E'
        elif asc == 5:
            asc = 'J'
        elif asc == 6:
            asc = 'K'
        print("Computer guessed: " + asc)
        ##continue
        if letter == temp[1]:
            numRight = numRight + 1
        else:
            print('Result was wrong, computers second choice:')
            temp = sorted(final)
            temp.pop()
            temp = temp.pop()
            print(temp)
            asc = temp[1]
            if asc == 0:
                asc = 'A'
            elif asc == 1:
                asc = 'B'
            elif asc == 2:
                asc = 'C'
            elif asc == 3:
                asc = 'D'
            elif asc == 4:
                asc = 'E'
            elif asc == 5:
                asc = 'J'
            elif asc == 6:
                asc = 'K'
            print("Computer guessed: " + asc)
        print('---------------------------')
        chars = []
        count = 0
        if letter < 6:
            letter = letter +1
        else:
            letter = 0
        total = total + 1

print("accuracy:" + str((numRight/total)*100))
