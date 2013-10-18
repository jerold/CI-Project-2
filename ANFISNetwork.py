#!/usr/bin/python
# RBF Network

import json
import random
import sys
import math
import time

eta = 1.00

logWeights = False
logResults = False
logError = True

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
    Input, Rules, ProdNorm, Consequent, Summation, Output= range(6)

    @classmethod
    def desc(self, x):
        return {
            self.Input:"I",
            self.Rules:"R",
            self.ProdNorm:"P",
            self.Consequent:"C",
            self.Summation:"H",
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
    if isinstance(p, list):
        for i in range(len(p)):
            sumOfSquares = sumOfSquares + ((p[i]-q[i])*(p[i]-q[i]))
    else:
        sumOfSquares = sumOfSquares + ((p-q)*(p-q))
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

# values will be a vector, we find k centers among them
def kMeans(values, k):
    centers = random.sample(values, k)
    memberships = []
    movement = 100.0
    while abs(movement) > 0.0:
        movement = 0.0
        memberships = []
        for k, v in enumerate(centers):
            memberships.append([])
        # establish each value's membership to one of the k centers
        for i, v in enumerate(values):
            bestDist = 99999
            bestCenter = 0
            for j, c in enumerate(centers):
                dist = euclidianDistance(v, c)
                if dist < bestDist:
                    bestDist = dist
                    bestCenter = j
            memberships[bestCenter].append(i)
        # avg centers' members for new center location
        for i, center in enumerate(centers):
            if len(memberships[i]) > 0:
                newCenter = 0.0
                for j, v in enumerate(memberships[i]):
                    newCenter = newCenter + values[v]
                newCenter = newCenter/len(memberships[i])
                movement = movement + (center - newCenter)
                centers[i] = newCenter
    # Construct Sigma for the rules
    sigmas = []
    for i, center in enumerate(centers):
        sigmas.append(0.0)
        for j, v in enumerate(memberships[i]):
            sigmas[i] = sigmas[i] + (values[v]-center)*(values[v]-center)
        sigmas[i] = math.sqrt(1.0/len(memberships[i])*sigmas[i])
    return {'centers':centers,
            'members':memberships,
            'sigmas':sigmas}
            

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

# A Pattern set contains sets of 3 types of patterns
# and can be used to retrieve only those patterns of a certain type
class PatternSet:
    # patterns = []
    # inputMagX = 1
    # inputMagY = 1
    # outputMag = 0
    # centers = {}
    # confusionMatrix = {}
    # targetMatrix = {}

    # Reads patterns in from a file, and puts them in their coorisponding set
    def __init__(self, fileName):
        with open(fileName) as jsonData:
            data = json.load(jsonData)
            
        # Assign Patterns and Randomize order
        self.patterns = data['patterns']
        self.inputMagX = len(self.patterns[0]['p'])
        self.inputMagY = 1
        if isinstance(self.patterns[0]['p'][0], list):
            self.inputMagX = len(self.patterns[0]['p'][0])
            self.inputMagY = len(self.patterns[0]['p'])

        random.shuffle(self.patterns)
        print(str(len(self.patterns)) + " Patterns Available (" + str(self.inputMagY) + "x" + str(self.inputMagX) + ")")

        # Assign Centers
        self.centers = data['centers']
        self.sigmas = data['sigmas']

        # Architecture has 1 output node for each digit / letter
        # Currently this also corrisponds to each center
        # Assemble our target and confusion matrix
        keys = list(self.centers.keys())
        keys.sort()
        print("Centers: [" + ', '.join(str(k).split('.')[0] for k in keys) + "]")
        self.confusionMatrix = {}
        self.targetMatrix = {}
        index = 0
        # Initialize Confusion Matrix and Target Matrix
        for key in keys:
            self.confusionMatrix[key] = [0.0] * len(keys)
            self.targetMatrix[key] = [0] * len(keys)
            self.targetMatrix[key][index] = 1
            index = index + 1
        self.outputMag = len(keys)

        # Sigma = (max euclidian distance between all centers) / number of centers
        maxEuclidianDistance = 0.0
        for k1 in keys:
            for k2 in keys:
                maxEuclidianDistance = max(euclidianDistance(vectorizeMatrix(self.centers[k1]), vectorizeMatrix(self.centers[k2])), maxEuclidianDistance)
        Neuron.sigma = maxEuclidianDistance/math.sqrt(len(keys))
        print("Sigma: " + str(Neuron.sigma))

    def printConfusionMatrix(self):
        keys = list(self.confusionMatrix.keys())
        keys.sort()
        for key in keys:
            printPatterns(self.confusionMatrix[key])

    def targetVector(self, key):
        return self.targetMatrix[str(key)]

    def updateConfusionMatrix(self, key, outputs):
        maxIndex = 0
        maxValue = 0
        for i in range(len(outputs)):
            if maxValue < outputs[i]:
                maxIndex = i
                maxValue = outputs[i]
        # print("Key: " + str(key) + " winner:" + str(maxIndex))
        self.confusionMatrix[str(key)][maxIndex] = self.confusionMatrix[str(key)][maxIndex] + 1

    def inputMagnitude(self):
        return self.inputMagX * self.inputMagY

    def outputMagnitude(self):
        return self.outputMag


class Net:
    def __init__(self, patternSet):
        inputLayer = Layer(NetLayerType.Input, None, patternSet.inputMagnitude())
        ruleLayer = Layer(NetLayerType.Rules, inputLayer, patternSet.inputMagnitude())
        prodNormLayer = Layer(NetLayerType.ProdNorm, ruleLayer, patternSet.outputMagnitude())
        consequentLayer = Layer(NetLayerType.Consequent, prodNormLayer, patternSet.outputMagnitude())
        consequentLayer.consequences = [randomInitialWeight()] * patternSet.inputMagnitude() + [randomInitialWeight()]
        consequentLayer.consequences = consequentLayer.consequences * len(prodNormLayer.neurons)
        summationLayer = Layer(NetLayerType.Output, consequentLayer, patternSet.outputMagnitude())
        self.layers = [inputLayer, ruleLayer, prodNormLayer, consequentLayer, summationLayer]
        self.patternSet = patternSet
        self.absError = 100
        if self.patternSet.inputMagY > 1:
            self.buildCenters()
        else:
            self.buildRules()

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
                self.layers[NetLayerType.Consequent].adjustWeights(self.patternSet.targetVector(patterns[i]['t']))
            else:
                self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[NetLayerType.Consequent].getOutputs())
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

    def buildRules(self):
        #sort training patterns by unique targets
        keys = list(self.patternSet.centers.keys())
        keys.sort()
        # columns in the patterns represent attributes, build attribute array
        attributes = []
        for i in range(len(self.patternSet.centers[keys[0]])):
            attributes.append([])
        for key in keys:
            pattern = self.patternSet.centers[key]
            for i, attribute in enumerate(pattern):
                attributes[i].append(attribute)
        # Now we build a center for each attribute
        centers = []
        for k, v in enumerate(attributes):
            centerCount = 1
            attributeCenters = kMeans(v, centerCount)
            increaseCount = True
            while increaseCount:
                centerCount = centerCount + 1
                newAttributeCenters = kMeans(v, centerCount)
                if 0.0 in newAttributeCenters['sigmas']:
                    increaseCount = False
                else:
                    attributeCenters = newAttributeCenters
            centers.append(attributeCenters)
        # Print Final Rules
        for a, attributeDetails in enumerate(centers):
            for c, center in enumerate(attributeDetails['centers']):
                memString = ', '.join(str(round(x, 3)) for x in attributeDetails['members'][c])
                print("C:" + str(a) + ":" + str(c) + "[" + str(round(center, 2)) + "{" + str(round(attributeDetails['sigmas'][c], 2)) + "}]  (" + memString + ")")
            print("\n")

        # build the rulesets filling them out with rules corresponding to the centers assembled above
        # link up the product nodes to their corresponding rulesets' rules
        ruleLayer = self.layers[NetLayerType.Rules]
        prodLayer = self.layers[NetLayerType.ProdNorm]
        for i, attribute in enumerate(attributes):
            newRuleSet = RuleSet(ruleLayer)
            for j, center in enumerate(centers[i]['centers']):
                neuron = Neuron(ruleLayer)
                neuron.center = center
                neuron.sigma = centers[i]['sigmas'][j]
                newRuleSet.rules.append(neuron)
                for member in centers[i]['members'][j]:
                    prodLayer.neurons[member].inputVector.append(j)
            ruleLayer.ruleSets.append(newRuleSet)
        print(ruleLayer)
        print(prodLayer)

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
        self.ruleSets =[]
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

    # the product layer will request specific rule outputs from the ruleset layer
    def getRuleLayerOutputs(self, inputVector):
        outputs = []
        for rIndex, ruleSet in enumerate(self.ruleSets):
            print("IV:" + str(inputVector[rIndex]) + " RS:" + str(len(ruleSet.rules)))
            if inputVector[rIndex] > len(ruleSet.rules):
                raise NameError('Rule Index in InputVector does not match the number of Rules in this Ruleset!')
            outputs.append(ruleSet.rules[inputVector[rIndex]].output)
        return outputs

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

    def adjustConsequent(self, targets):
        if len(targets) != len(self.neurons):
            raise NameError('Output dimension of network does not match that of target!')
        for i, neuron in enumerate(self.neurons):
            error = abs(targets[i] - neuron.output)
            while error > 0.05:
                for j, consequent in enumerate(self.consequences):
                    self.consequences[j] = consequent + (eta * (targets[i] - neuron.output) * consequent)
                self.feedForward()
                error = abs(targets[i] - neuron.output)

    # Each Layer has a link to the next link in order.  Input values are translated from
    # input to output in keeping with the Layer's function
    def feedForward(self):
        if self.layerType == NetLayerType.Input:
            # Input Layer feeds all input to output with no work done
            for neuron in self.neurons:
                neuron.output = neuron.input
            self.next.feedForward()
        elif self.layerType == NetLayerType.Rules:
            prevOutputs = self.prev.getOutputs()
            for rsIndex, ruleSet in enumerate(self.ruleSets):
                for rIndex, rule in enumerate(ruleSet.rules):
                    # ANFIS on the Euclidian Norm of input to rule center
                    rule.input = prevOutputs[rsIndex]
                    rule.input = euclidianDistance(prevOutputs[rsIndex], rule.center);
                    rule.output = radialBasisFunction(rule.input, rule.sigma)
            self.next.feedForward()
        elif self.layerType == NetLayerType.ProdNorm:
            # take product
            rollingSum = 0.0
            for neuron in self.neurons:
                prevOutputs = self.prev.getRuleLayerOutputs(neuron.inputVector)
                neuron.input = prevOutputs[0]
                for outputValue in prevOutputs[1:]:
                    neuron.input = neuron.input*outputValue
                rollingSum = rollingSum + neuron.input
            # Normalize output
            print("Outputs")
            if abs(rollingSum) > 0.0:
                for neuron in self.neurons:
                    neuron.output = neuron.input/rollingSum
                    print(round(neuron.output, 3))
            self.next.feedForward()
        elif self.layerType == NetLayerType.Consequent:
            prevOutputs = self.prev.getOutputs()
            for i, neuron in enumerate(self.neurons):
                consequenceSum = 0.0
                for j, val in enumerate(prevOutputs):
                    consequenceSum += val*self.consequences[i][j]
                consequenceSum += self.consequences[-1]
                neuron.output = consequenceSum
        elif self.layerType == NetLayerType.Summation:
            raise("Balls")
        elif self.layerType == NetLayerType.Output:
            raise("Balls")


            
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
        out = "" + NetLayerType.desc(self.layerType) + "[\n"
        if self.layerType == NetLayerType.Rules:
            for rs, attribute in enumerate(self.ruleSets):
                out = out + "  A:" + str(rs) + "["
                for rule in attribute.rules:
                    out = out + "(" + str(round(rule.center, 2)) + ":" + str(round(rule.sigma, 2)) + ")"
                out = out + "]\n"
        if self.layerType == NetLayerType.ProdNorm:
            for p, prodNode in enumerate(self.neurons):
                out = out + "  P:" + str(p) + "["
                for r, ruleIndex in enumerate(prodNode.inputVector):
                    out = out + "(" + str(r) + ":" + str(ruleIndex) + ")"
                out = out + "]\n"
        out = out + "]\n"
        return out


class RuleSet:
    def __init__(self, layer):
        self.layer = layer
        self.rules = []

    def setInputs(self, iput):
        for neuron in self.rules:
            neuron.input = iput


# Neuron contains inputs and outputs and depending on the type will use
# weights or centers in calculating it's outputs.  Calculations are done
# in the layer as function of the neuron is tied to the layer it is contained in
class Neuron:
    def __init__(self, layer):
        self.layer = layer
        self.input = 0.00
        self.output = 0.00
        self.inputVector = []
        self.center = []
        self.sigma = 0.00
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
    #p = PatternSet('data/optdigits/optdigits-orig.json')   # 32x32
    #p = PatternSet('data/letter/letter-recognition.json')  # 1x16 # Try 1 center per attribute, and allow outputs to combine them
    p = PatternSet('data/pendigits/pendigits.json')        # 1x16 # same as above
    #p = PatternSet('data/semeion/semeion.json')            # 16x16 # Training set is very limited
    #p = PatternSet('data/optdigits/optdigits.json')        # 8x8
    #for e in range(1, 20):
    n = Net(p)
    n.run(PatternType.Train, 0, 10000)
    n.run(PatternType.Test, 10000, 10992)

    p.printConfusionMatrix()
    print("Done")
