#!/usr/bin/python
# RBF Network

import sys
import random
import math
import time
import patternSet
from patternSet import PatternSet

eta = 1.00

# Enum for Pattern Type ( Also used as Net running Mode)
class PatternType:
    Train, Test, Validate = range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Train:"Train",
            self.Test:"Test",
            self.Validate:"Validate"}[x]

# Enum for Layer Type
class NetLayerType:
    Input, Hidden, Output = range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Input:"I",
            self.Hidden:"H",
            self.Output:"O"}[x]

# Weights are initialized to a random value between -0.3 and 0.3
def randomInitialWeight():
    return float(random.randrange(0, 6001))/10000 - .3

# RBF used in Hidden Layer output calculation
def radialBasisFunction(norm, sigma):
    #Inverse Multiquadratic
    return 1.0/math.sqrt(norm*norm + sigma*sigma)

# used in calculating Sigma based on center locations
def euclidianDistance(p, q):
    sumOfSquares = 0.0
    for i in range(len(p)):
        sumOfSquares = sumOfSquares + ((p[i]-q[i])*(p[i]-q[i]))
    return math.sqrt(sumOfSquares)

# combined sum of the difference between two vectors
def outputError(p, q):
    errSum = 0.0
    for i in range(len(p)):
        errSum = errSum + math.fabs(p[i] - q[i])
    return errSum

# Combination of two vectors
def linearCombination(p, q):
    lSum = 0.0
    for i in range(len(p)):
        lSum = lSum + p[i]*q[i]
    return lSum

def vectorizeMatrix(p):
    if isinstance(p[0], list):
        v = []
        for i in p:
            v = v + i
        return v
    else:
        return p

# Time saver right here
def clearLogs():
    with open('errors.txt', 'w') as file:
        file.truncate()
    with open('results.txt', 'w') as file:
        file.truncate()
    with open('weights.txt', 'w') as file:
        file.truncate()

# print an individual pattern with or without target value
def printPatterns(pattern):
    if isinstance(pattern, dict):
        for key in pattern.keys():
            if key == 't':
                print("Target: " + str(key))
            elif key == 'p':
                printPatterns(pattern['p'])
    elif isinstance(pattern[0], list):
        for pat in pattern:
            printPatterns(pat)
    else:
        print(', '.join(str(round(x, 3)) for x in pattern))


class Net:
    def __init__(self, patternSet):
        inputLayer = Layer(NetLayerType.Input, None, patternSet.inputMagnitude())
        hiddenLayer = Layer(NetLayerType.Hidden, inputLayer, patternSet.outputMagnitude())
        outputLayer = Layer(NetLayerType.Output, hiddenLayer, patternSet.outputMagnitude())
        self.layers = [inputLayer, hiddenLayer, outputLayer]
        self.patternSet = patternSet
        self.absError = 100
        self.buildCenters()

    # Run is where the magic happens. Training Testing or Validation mode is indicated and
    # the coorisponding pattern set is loaded and ran through the network
    # At the end Error is calculated
    def run(self, mode, startIndex, endIndex):
        patterns = self.patternSet.patterns
        eta = 1.0
        errorSum = 0.0
        print("Mode[" + PatternType.desc(mode) + ":" + str(endIndex - startIndex) + "]")
        startTime = time.time()
        for i in range(startIndex, endIndex):
            #Initialize the input layer with input values from the pattern
            # Feed those values forward through the remaining layers, linked list style
            self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
            self.layers[NetLayerType.Input].feedForward()
            if mode == PatternType.Train:
                #For training the final output weights are adjusted to correct for error from target
                self.layers[NetLayerType.Output].adjustWeights(self.patternSet.targetVector(patterns[i]['t']))
            else:
                self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[NetLayerType.Output].getOutputs())
                # print("Output:")
                # printPatterns(self.layers[NetLayerType.Output].getOutputs())
                # print("Target:")
                # printPatterns(self.patternSet.targetVector(patterns[i]['t']))
            # Each pattern produces an error which is added to the total error for the set
            # and used later in the Absolute Error Calculation
            outError = outputError(self.layers[NetLayerType.Output].getOutputs(), self.patternSet.targetVector(patterns[i]['t']))
            errorSum = errorSum + outError
            eta = eta - eta/((endIndex - startIndex)*1.1)
            # if mode != PatternType.Train and logResults:
            #     # Logging
            #     with open('results.txt', 'a') as file:
            #         out = ""
            #         for output in self.layers[NetLayerType.Output].getOutputs():
            #             out = out + str(round(output, 2)) + '\t'
            #         for target in patterns[i]["outputs"]:
            #             out = out + str(round(target, 2)) + '\t'
            #         file.write(out + '\n')
            # self.recordWeights()
        endTime = time.time()
        print("Time [" + str(round(endTime-startTime, 4)) + "sec]")
        # if mode != PatternType.Train:
        #     # Calculate Absolute Error pg.398
        #     self.absError = 1.0/(patCount*len(patterns[0]["outputs"]))*errorSum
        #     print("Absolute Error: " + str(round(self.absError, 4)) + " [" + str(endTime-startTime) + "]")
        #     if logError:
        #         # Logging
        #         with open('errors.txt', 'a') as file:
        #             file.write(str(round(self.absError, 4)) + '\t' + str(endTime-startTime) + '\n')

    # During this process we calculate sigma which is used in the Hidden Layers' RBF function
    def buildCenters(self):
        centers = self.patternSet.centers
        neurons = self.layers[NetLayerType.Hidden].neurons
        n = 0
        maxEuclidianDistance = 0.0
        # print("Centers:")
        keys = list(centers.keys())
        keys.sort()
        for key in keys:
            neurons[n].center = vectorizeMatrix(centers[key])
            n = n + 1
        # for n in range(len(centers)):
        #     printPatterns(neurons[n].center)

    # Logging
    def recordWeights(self):
        self.logWeightIterator = self.logWeightIterator + 1
        if logWeights and self.logWeightIterator%self.logWeightFrequency == 0:
            with open('weights.txt', 'a') as file:
                out = ""
                for neuron in self.layers[NetLayerType.Output].neurons:
                    for weight in neuron.weights:
                        out = out + str(round(weight, 2)) + '\t'
                file.write(out + '\n')        

    # Output Format
    def __str__(self):
        out = "N[\n"
        for layer in self.layers:
            out = out + str(layer)
        out = out + "]\n"
        return out

#Layers are of types Input Hidden and Output.  
class Layer:
    def __init__(self, layerType, prevLayer, neuronCount):
        self.layerType = layerType
        self.prev = prevLayer
        if prevLayer != None:
            prevLayer.next = self
        self.next = None
        self.neurons = []
        for n in range(neuronCount):
            self.neurons.append(Neuron(self))

    # Assign input values to the layer's neuron inputs
    def setInputs(self, inputVector):
        if len(inputVector) != len(self.neurons):
            raise NameError('Input dimension of network does not match that of pattern!')
        for p in range(len(self.neurons)):
            self.neurons[p].input = inputVector[p]

    #return a vector of this Layer's Neuron outputs
    def getOutputs(self):
        out = []
        for neuron in self.neurons:
            out.append(neuron.output)
        return out

    # Adjusting weights is done on the output layer in order to scale the
    # output of a neuron's RBF function.
    def adjustWeights(self, targets):
        if len(targets) != len(self.neurons):
            raise NameError('Output dimension of network does not match that of target!')
        # DeltaWkj = (learningRate)sum(TARGETkp - OUTPUTkp)Yjp
        prevOutputs = self.prev.getOutputs()
        # print("O:" + str(round(self.neurons[0].output, 2)) + "  T:" + str(round(targets[0], 2)))
        for k in range(len(self.neurons)):
            neuron = self.neurons[k]
            for j in range(len(prevOutputs)):
                neuron.weightDeltas[j] = eta * (targets[k] - neuron.output) * prevOutputs[j]
                neuron.weights[j] = neuron.weights[j] + neuron.weightDeltas[j]
                if neuron.weights[j] > 9999999:
                    raise NameError('Divergent Weights!')

    # Each Layer has a link to the next link in order.  Input values are translated from
    # input to output in keeping with the Layer's function
    def feedForward(self):
        if self.layerType == NetLayerType.Input:
            # Input Layer feeds all input to output with no work done
            for neuron in self.neurons:
                neuron.output = neuron.input
            self.next.feedForward()
        elif self.layerType == NetLayerType.Hidden:
            # RBF on the Euclidian Norm of input to center
            for neuron in self.neurons:
                prevOutputs = self.prev.getOutputs()
                if len(neuron.center) != len(prevOutputs):
                    raise NameError('Center dimension does not match that of previous Layer outputs!')
                neuron.input = euclidianDistance(prevOutputs, neuron.center);
                neuron.output = radialBasisFunction(neuron.input, Neuron.sigma)
            self.next.feedForward()
        elif self.layerType == NetLayerType.Output:
            # Linear Combination of Hidden layer outputs and associated weights
            for neuron in self.neurons:
                prevOutputs = self.prev.getOutputs()
                if len(neuron.weights) != len(prevOutputs):
                    raise NameError('Weights dimension does not match that of previous Layer outputs!')
                neuron.output = linearCombination(prevOutputs, neuron.weights)

    # Output Format
    def __str__(self):
        out = "  " + NetLayerType.desc(self.layerType) + "["
        for neuron in self.neurons:
            out = out + str(neuron)
        out = out + "]\n"
        return out

# Neuron contains inputs and outputs and depending on the type will use
# weights or centers in calculating it's outputs.  Calculations are done
# in the layer as function of the neuron is tied to the layer it is contained in
class Neuron:
    sigma = 0.0
    
    def __init__(self, layer):
        self.layer = layer
        self.input = 0.00
        self.output = 0.00
        self.center = []
        self.weights = []
        self.weightDeltas = []
        if layer.prev != None:
            for w in range(len(layer.prev.neurons)):
                self.weights.append(randomInitialWeight())
                self.weightDeltas.append(0.0)

    # Output Format
    def __str__(self):
        out = "{" + str(round(self.input,2)) + "["
        if self.layer.layerType == NetLayerType.Output:
            for w in self.weights:
                out = out + str(round(w,2)) + ","
        elif self.layer.layerType == NetLayerType.Hidden:
            for c in self.center:
                out = out + str(round(c,2)) + ","
        out = out + "]" + str(round(self.output,2)) + "} "
        return out

#Main
if __name__=="__main__":
    trainPercentage = 0.8
    #p = PatternSet('data/optdigits/optdigits-orig.json', trainPercentage)   # 32x32
    #p = PatternSet('data/letter/letter-recognition.json', trainPercentage)  # 20000 @ 1x16 # Try 1 center per attribute, and allow outputs to combine them
    #p = PatternSet('data/pendigits/pendigits.json', trainPercentage)        # 10992 @ 1x16 # same as above
    #p = PatternSet('data/semeion/semeion.json', trainPercentage)           # 1593 @ 16x16 # Training set is very limited
    p = PatternSet('data/optdigits/optdigits.json', trainPercentage)        # 5620 @ 8x8
    
    n = Net(p)
    n.run(PatternType.Train, 0, int(p.count*trainPercentage))
    n.run(PatternType.Test, int(p.count*trainPercentage), p.count)

    p.printConfusionMatrix()
    print("Done")
