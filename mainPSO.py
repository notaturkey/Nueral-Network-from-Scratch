import random
import copy
import math

inner_nodes = 30
output_nodes = 7

particles_const = 10
epochs = 10
c1 = 0.98
c2 = 0.60
w = 0.3
min_velocity = -100
max_velocity = 100

##sigmoids 
def bipolarSigmoid(x):
    return (2 / (1 + math.exp(-1*x ))) - 1

##swarm for pso
class Swarm():
    def __init__(self, particles):
        self.particles = particles
        self.gbestScore = -1000
        self.gbest = []

    def update(self):
        scores = []
        for part in self.particles:
            scores.append([part.score, part])
        
        scores = sorted(scores, key=lambda score: score[0])

        for i in scores:
            if i[0] > self.gbestScore:
                self.gbestScore = i[0]
                self.gbest = i[1].velocity
            if i[0] > i[1].pbestScore:
                i[1].pbestScore = i[0]
                i[1].pbest = i[1].velocity 

        
        for part in self.particles:
            index = 0
            part.nxtVelocity = []
            for i in part.velocity:
                temp = (w*i) + (c1*random.random()*(part.pbest[index] - i))
                temp = temp + (c2*random.random()*(self.gbest[index] - i))
                index = index + 1
                part.nxtVelocity.append(temp)

            ind = 0
            newState = []
            for n in part.nxtVelocity:
                if n>max_velocity:
                    n = max_velocity
                elif n<min_velocity:
                    n= min_velocity
                ind = ind +1
                newState.append(n)
            
            part.velocity = newState


    

##particle for pso
class Particle():
    def __init__ (self):
        self.score = 0
        self.nxtVelocity = []
        self.velocity = self.initVel()
        self.pbest = self.velocity
        self.pbestScore = -1000
        self.gbest = []

    def initVel(self):
        temp = (63*inner_nodes) + (inner_nodes*output_nodes)
        arr = []
        for i in range(temp):
            arr.append(random.uniform(-1,1))
        
        ##biases
        arr.append(random.uniform(-1,1))
        arr.append(random.uniform(-1,1))
        return arr
    
    def checkScore(self, result,target):
        score = 0
        for i in range(len(result)):
            score = score + (target[i] - result[i].signal)    
        self.score = abs(score)
    

##node object for net
class Node():
    def __init__ (self):
        self.signal = 0
        
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
            medianNodes.append(node)

        ##input layer
        inputNodes = []
        for i in range(63):
            node = Node()
            inputNodes.append(node)
        
        self.net.append(inputNodes)
        self.net.append(medianNodes)
        self.net.append(outputNodes)
    
    def feed(self, chars, particle):
        count = 0
        for i in chars:
            if i != '.':
                self.net[0][count].signal = 1
            else:
                self.net[0][count].signal = -1
            count = count+1

        self.feedForward(particle)

    def feedForward(self, particle):
        temp = 0
        weights = []
        ##getting right weights from particle
        for j in self.net[0]:
            k = particle.velocity[(inner_nodes*temp):(inner_nodes*(temp+1))]
            weights.append(k)
            temp = temp+1
        
        tempArr = particle.velocity[63*inner_nodes:]
        arr = []
        temp = 0
        for j in self.net[1]:
            k = tempArr[(output_nodes*temp):(output_nodes*(temp+1))]
            arr.append(k)
            temp = temp+1

        count = 0
        ##first layer
        for i in self.net[1]:
            sum = 0
            temp = 0
            for j in self.net[0]:
                sum = sum + (j.signal*weights[temp][count])
                temp = temp+1
            sum = particle.velocity[-2] + sum
            i.signal = bipolarSigmoid(sum)
            count = count + 1

        ##second layer
        count = 0
        for i in self.net[2]:
            sum = 0
            temp = 0
            for j in self.net[1]:
                sum = sum + (j.signal*arr[temp][count])
                temp = temp+1
            sum = particle.velocity[-1] + sum
            i.signal = bipolarSigmoid(sum)
            count = count + 1


netw = Net()
netw.buildNet()
f = open('HW3_Training.txt', 'r')
target = [[1,-1,-1,-1,-1, -1, -1],[-1,1,-1,-1,-1, -1, -1],[-1,-1, 1,-1,-1, -1, -1],[-1,-1,-1, 1,-1, -1, -1],[-1,-1,-1,-1, 1, -1, -1],[-1,-1,-1,-1,-1, 1, -1],[-1,-1,-1,-1,-1, -1, 1]] 
particles = []
for i in range(particles_const):
        part = Particle()
        particles.append(part)

swarm = Swarm(particles)
def train(netw,swarm):
    for i in range(100):
        for part in swarm.particles:
            part.score = 0
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
                if letter < 7:
                    for part in swarm.particles:
                        netw.feed(chars,part)
                        #part.checkScore(netw.net[2],target[letter])
                        temp = copy.deepcopy(netw.net[2])
                        final = []
                        count  = 0
                        for i in temp:
                            final.append([i.signal,count])
                            count = count+1
                        temp = sorted(final).pop()
                        asc = temp[1]
                        if asc == letter:
                            part.score = part.score+1
                        
                    letter = letter + 1
                else:
                    letter = 0
                    for part in swarm.particles:
                        netw.feed(chars,part)
                        #part.checkScore(netw.net[2],target[letter])
                        temp = copy.deepcopy(netw.net[2])
                        final = []
                        count  = 0
                        for i in temp:
                            final.append([i.signal,count])
                            count = count+1
                        temp = sorted(final).pop()
                        asc = temp[1]
                        if asc == letter:
                            part.score = part.score+1
            
                    letter = letter + 1
                ##continue
                chars = []
                count = 0

        swarm.update()

        f.seek(0)

print('swarming particles, may take a bit')
train(netw,swarm)
f.close()

print('particle scores from final run:')
for i in swarm.particles:
    print(str(i.score/21) + " percent accurate")

print('global best accuracy: '+str(swarm.gbestScore/21))

print('Testing with global best performing particle')
bestParticle = Particle()
bestParticle.velocity = swarm.gbest
##test
f = open('HW3_Testing.txt', 'r')
avgTotal = 0 
for runs in range(10):
    netw = Net()
    netw.buildNet()
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
    avgTotal = avgTotal+ (numRight/total)*100
    f.seek(0)
    runs = runs + 1
print("Average accuracy:")
print(avgTotal/10)

f.close()
