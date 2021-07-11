import random
import copy
import math

inner_nodes = 15
output_nodes = 7

particles_const = 20
epochs = 100
c1 = 0.6
c2 = 1.95
w = 0.00051231
min_velocity = -10
max_velocity = 10

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
        for part in self.particles:
            if part.score > self.gbestScore:
                self.gbestScore = part.score
                self.gbest = part.pos
            if part.score > part.pbestScore:
                part.pbestScore = part.score
                part.pbest = part.pos

            part.nxtVelocity = []
            for i in range(len(part.velocity)):
                temp = (w*part.velocity[i]) + (c1*random.uniform(-1,1)*(part.pbest[i] - part.pos[i]))
                temp = temp + (c2*random.uniform(-1,1)*(self.gbest[i] - part.pos[i]))
                if temp > max_velocity:
                    temp = max_velocity
                    part.nxtVelocity.append(temp)
                elif temp < min_velocity:
                    temp = min_velocity
                    part.nxtVelocity.append(temp)
                else:
                    part.nxtVelocity.append(temp)


            #update velocity
            for i in range(len(part.velocity)):
                    if part.pos[i]+ part.nxtVelocity[i] > 3:
                        part.pos[i] = 3
                    elif part.pos[i]+ part.nxtVelocity[i] < -3:
                        part.pos[i] = -3
                    else:
                        part.pos[i] = part.pos[i]+ part.nxtVelocity[i]





##particle for pso
class Particle():
    def __init__ (self):
        self.score = 0
        self.pos = self.initPos()
        self.nxtVelocity = []
        self.velocity = self.initVel()
        self.pbest = self.velocity
        self.pbestScore = -1000

    def initPos(self):
        temp = (63*inner_nodes) + (inner_nodes*output_nodes)
        arr = []
        for i in range(temp):
            arr.append(random.uniform(-2,2))

        ##biases
        arr.append(10)
        arr.append(5)
        return arr

    def initVel(self):
        temp = (63*inner_nodes) + (inner_nodes*output_nodes)
        arr = []
        for i in range(temp):
            arr.append(random.uniform(min_velocity,max_velocity))

        ##biases
        arr.append(random.uniform(-1,1))
        arr.append(random.uniform(-1,1))
        return arr


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
            k = particle.pos[(inner_nodes*temp):(inner_nodes*(temp+1))]
            weights.append(k)
            temp = temp+1

        tempArr = particle.pos[63*inner_nodes:]
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
            sum = particle.pos[-2] + sum
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
            sum = particle.pos[-1] + sum
            i.signal = bipolarSigmoid(sum)
            count = count + 1


target = [[1,-1,-1,-1,-1, -1, -1],[-1,1,-1,-1,-1, -1, -1],[-1,-1, 1,-1,-1, -1, -1],[-1,-1,-1, 1,-1, -1, -1],[-1,-1,-1,-1, 1, -1, -1],[-1,-1,-1,-1,-1, 1, -1],[-1,-1,-1,-1,-1, -1, 1]]
def train(netw,swarm):
    f = open('TrainData.txt', 'r')
    for i in range(epochs):
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
    f.close()


##test
f = open('TestData.txt', 'r')
avgTotal = 0
for runs in range(10):
    particles = []
    for i in range(particles_const):
        part = Particle()
        particles.append(part)

    swarm = Swarm(particles)
    netw = Net()
    netw.buildNet()
    print('swarming particles, may take a bit depending on # of epochs')
    train(netw,swarm)


    print('particle scores from final run:')
    for i in swarm.particles:
        print(str(i.score/21) + " percent accurate")

    print('global best accuracy: '+str(swarm.gbestScore/21))

    print('Testing with global best performing particle')
    bestParticle = Particle()
    bestParticle.pos = swarm.gbest
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
            netw.feed(chars, bestParticle)
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
