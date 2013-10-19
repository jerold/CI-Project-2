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
    if abs(sigma) > 0.0:
        return 1.0/math.sqrt(norm*norm + sigma*sigma)
    else:
        return 1.0

# used in calculating Sigma based on center locations
def euclidianDistance(p, q):
    sumOfSquares = 0.0
    if isinstance(p, list):
        if isinstance(p[0], list):
            for i in range(len(p)):
                for j in range(len(p[i])):
                    sumOfSquares = sumOfSquares + ((p[i][j]-q[i][j])*(p[i][j]-q[i][j]))
        else:
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
    #print(values)
    for i, center in enumerate(centers):
        sigmas.append(0.0)
        for j, v in enumerate(memberships[i]):
            sigmas[i] = sigmas[i] + (values[v]-center)*(values[v]-center)
        if abs(sigmas[i]) > 0.0:
            sigmas[i] = math.sqrt(1.0/len(memberships[i])*sigmas[i])
    return {'centers':centers,
            'members':memberships,
            'sigmas':sigmas}


# Centers are built for each of the targets in two steps
# First an average is built for each target from each pattern of that target type
# Next we run k-means on the new centers again all patterns
# Sigmas are calculated for each element
def buildCentersAndSigmas(patterns):
    centersTargets = {}
    for pattern in patterns:
        if pattern['t'] not in centersTargets:
            centersTargets[pattern['t']] = []
        centersTargets[pattern['t']].append(pattern)
    centers = {}
    sigmas = {}
    print("Found " + str(len(centersTargets)) + " targets.")
    print("Constructing Centers and Sigmas...")
    # build center as mean of all trained k patterns, and sigma as standard deviation
    for k in centersTargets.keys():
        kPats = centersTargets[k]
        centers[k] = buildMeanPattern(kPats)

    # print("Centers Post Average:")
    # for k in centersTargets.keys():
    #     print(k)
    #     printPatterns(centers[k])

    # soften centers using k-means
    dist = 100
    distDelta = 100
    oldDist = 0
    while dist > 1 and abs(distDelta) > 0.01:
        tempCenters = adjustCenters(patterns, centers)
        dist = 0
        for k in centersTargets.keys():
            dist = dist + euclidianDistance(centers[k], tempCenters[k])
        centers = tempCenters
        distDelta = dist - oldDist
        oldDist = dist
        print("dist:" + str(round(dist, 4)) + ", delta:" + str(round(distDelta, 4)))

    # Build Sigmas for each space
    # print("Sigma:")
    # for k in centersTargets.keys():
    #     sigmas[k] = buildSigmaPattern(centers[k], kPats)
    #     printPatterns(sigmas[k])

    # print("Centers Post K-means:")
    # for k in centersTargets.keys():
    #     print(k)
    #     printPatterns(centers[k])

    return {'centers':centers, 'sigmas':sigmas}


def buildMeanPattern(patterns):
    h = 0
    w = len(patterns[0]['p'])
    if isinstance(patterns[0]['p'][0], list):
        h = len(patterns[0]['p'])
        w = len(patterns[0]['p'][0])
    mPat = emptyPattern(w, h)
    for pat in patterns:
        if h > 1:
            for i in range(h):
                for j in range(w):
                    mPat[i][j] = mPat[i][j] + pat['p'][i][j]
        else:
            for j in range(w):
                mPat[j] = mPat[j] + pat['p'][j]
    if h > 1:
        for i in range(h):
            for j in range(w):
                mPat[i][j] = mPat[i][j] / len(patterns)
    else:
        for j in range(w):
            mPat[j] = mPat[j] / len(patterns)
    return mPat


def buildSigmaPattern(meanPat, patterns):
    h = 0
    w = len(patterns[0]['p'])
    if isinstance(patterns[0]['p'][0], list):
        h = len(patterns[0]['p'])
        w = len(patterns[0]['p'][0])
    sPat = emptyPattern(w, h)
    # Sum over all square of distance from means
    if h > 1:
        for i in range(h):
            for j in range(w):
                for pat in patterns:
                    sPat[i][j] = sPat[i][j] + (pat['p'][i][j] - meanPat[i][j])*(pat['p'][i][j] - meanPat[i][j])
                sPat[i][j] = math.sqrt(1.0/len(patterns)*sPat[i][j])
    else:
        for j in range(w):
            for pat in patterns:
                sPat[j] = sPat[j] + (pat['p'][j] - meanPat[j])*(pat['p'][j] - meanPat[j])
            sPat[j] = math.sqrt(1.0/len(patterns)*sPat[j])
    return sPat


def adjustCenters(patterns, centers):
    groups = {}
    for k in centers.keys():
        groups[k] = []
    for pattern in patterns:
        bestDist = 99999
        bestKey = ''
        for key in centers.keys():
            center = centers[key]
            dist = euclidianDistance(pattern['p'], center)
            if dist < bestDist:
                bestDist = dist
                bestKey = key
        groups[bestKey].append(pattern)
    newCenters = {}
    for k in centers.keys():
        if len(groups[k]) > 0:
            newCenters[k] = buildMeanPattern(groups[k])
        else:
            newCenters[k] = centers[k]
    return newCenters


def emptyPattern(w, h):
    pat = []
    if h > 1:
        for i in range(h):
            pat.append([])
            for j in range(w):
                pat[i].append(0.0)
    else:
        for j in range(w):
            pat.append(0.0)
    return pat
            

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
    # Reads patterns in from a file, and puts them in their coorisponding set
    def __init__(self, fileName, percentTraining):
        with open(fileName) as jsonData:
            data = json.load(jsonData)
            
        # Assign Patterns and Randomize order
        self.patterns = data['patterns']
        self.count = data['count']
        self.inputMagX = len(self.patterns[0]['p'])
        self.inputMagY = 1
        if isinstance(self.patterns[0]['p'][0], list):
            self.inputMagX = len(self.patterns[0]['p'][0])
            self.inputMagY = len(self.patterns[0]['p'])

        random.shuffle(self.patterns)
        print(str(len(self.patterns)) + " Patterns Available (" + str(self.inputMagY) + "x" + str(self.inputMagX) + ")")

        # Construct Centers but base them only off the cases to be trained with
        centersAndSigmas = buildCentersAndSigmas(self.patterns[:int(data['count']*percentTraining)])
        self.centers = centersAndSigmas['centers']
        self.sigmas = centersAndSigmas['sigmas']

        # Architecture has 1 output node for each digit / letter
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

    def printConfusionMatrix(self):
        keys = list(self.confusionMatrix.keys())
        keys.sort()
        print("\nConfusion Matrix")
        for key in keys:
            printPatterns(self.confusionMatrix[key])
        print("\nKey, Precision, Recall")
        #for key in keys:
            #print(str(key) + ", " + str(round(self.calcPrecision(key), 3)) + ", " + str(round(self.calcRecall(key), 3)))
        self.calcPrecisionAndRecall()

    def calcPrecision(self, k):
        tp = self.confusionMatrix[k][k]
        fpSum = sum(self.confusionMatrix[k])
        if fpSum == 0.0:
            return fpSum
        return tp/fpSum

    def calcRecall(self, k):
        tp = self.confusionMatrix[k][k]
        keys = list(self.confusionMatrix.keys())
        keys.sort()
        i = 0
        tnSum = 0.0
        for key in keys:
            tnSum = tnSum + self.confusionMatrix[key][k]
        if tnSum == 0.0:
            return tnSum
        return tp/tnSum

    def calcPrecisionAndRecall(self):
        keys = list(self.confusionMatrix.keys())
        keys.sort()
        i = 0
        precision = []
        recall = []
        diagonal = []
        for key in keys:
            row = self.confusionMatrix[key]
            rowSum = 0
            for j, val in enumerate(row):
                if i==j:
                    diagonal.append(val)
                rowSum += val
                if len(recall) == j:
                    recall.append(val)
                else:
                    recall[j] = recall[j] + val
            precision.append(rowSum)
            i += 1
        for i, elem in enumerate(diagonal):
            if abs(precision[i]) > 0.0 and abs(recall[i]) > 0.0:
                print(str(keys[i]) + ", " + str(elem / precision[i]) + ", " + str(elem/recall[i]))
        
    def targetVector(self, key):
        return self.targetMatrix[key]

    def updateConfusionMatrix(self, key, outputs):
        maxIndex = 0
        maxValue = 0
        for i in range(len(outputs)):
            if maxValue < outputs[i]:
                maxIndex = i
                maxValue = outputs[i]
        self.confusionMatrix[key][maxIndex] = self.confusionMatrix[key][maxIndex] + 1

    def inputMagnitude(self):
        return self.inputMagX * self.inputMagY

    def outputMagnitude(self):
        return self.outputMag


class Net:
    "The Net class contains Layers which in turn contain rulesets (containing rules) or neurons."
    # The architecture is set up based on the pattern set we pass in
    # Pattern sets contain all input-output pairs, and centers for each unique target type "0-9" or "A-Z"
    # The last step in setting up the architecture of the network is to build rulesets for each input attribute
    # and rules based on the number of functional centers within each attribute
    # individual rules are then hooked up to the appropreate product layer to construct predicate logic for our targets
    # buildRules() explains this process in more detail
    def __init__(self, patternSet):
        inputLayer = Layer(NetLayerType.Input, None, patternSet.inputMagnitude())
        ruleLayer = Layer(NetLayerType.Rules, inputLayer, patternSet.inputMagnitude())
        prodNormLayer = Layer(NetLayerType.ProdNorm, ruleLayer, patternSet.outputMagnitude())
        consequentLayer = Layer(NetLayerType.Consequent, prodNormLayer, patternSet.outputMagnitude())

        consequentLayer.consequences = [[randomInitialWeight() for _ in range(patternSet.inputMagnitude() + 1)] for _ in range(len(prodNormLayer.neurons))]

        self.layers = [inputLayer, ruleLayer, prodNormLayer, consequentLayer]
        self.patternSet = patternSet
        self.absError = 100
        self.buildRules()

    # Run is where the magic happens. Training and Testing mode is indicated and
    # patterns in the indicated range are ran through the network
    # At the end Error is calculated
    def run(self, mode, startIndex, endIndex):
        "Patterns wthin the indicated range are fed through the network for training or testing purposes"
        patterns = self.patternSet.patterns
        if len(patterns) < endIndex:
            print(len(patterns))
            print(endIndex)
            raise NameError('Yo dawg, where you think I is gunna get that many patterns?')
        eta = 1.0
        errorSum = 0.0
        print("Mode[" + PatternType.desc(mode) + ":" + str(endIndex - startIndex) + "]...")
        startTime = time.time()
        for i in range(startIndex, endIndex):
            if len(vectorizeMatrix(patterns[i]['p'])) != len(self.layers[NetLayerType.Input].neurons):
                raise NameError('Input vector length does not match neuron count on input layer!')

            #Initialize the input layer with input values from the  current pattern
            # Feed those values forward through the remaining layers, linked list style
            self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
            self.layers[NetLayerType.Input].feedForward()

            if mode == PatternType.Train:
                #For training the consequent values are adjusted to correct for error from target
                self.layers[NetLayerType.Consequent].adjustConsequent(self.patternSet.targetVector(patterns[i]['t']))
            else:
                self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[NetLayerType.Consequent].getOutputs())
                #self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[NetLayerType.ProdNorm].getOutputs())
                # print("\nOutputs")
                # print("[" + ", ".join(str(int(x+0.2)) for x in self.layers[NetLayerType.ProdNorm].getOutputs()) + "]")
                # print("Target")
                # print(self.patternSet.targetVector(patterns[i]['t']))

            outError = outputError(self.layers[NetLayerType.Consequent].getOutputs(), self.patternSet.targetVector(patterns[i]['t']))
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

    def buildRules(self):
        "Each input represents an attribute, and re construct a unique ruleset for each attribute."
        # RuleSets consist of several rules, multiple targets may share a single rule within a Ruleset
        # ie. "1" and "7" may have similar widths, and therefore have a similar center in the width attribute
        # Each target is a member of one rule within the ruleset, it's membership is measured along a gaussian curve

        #sort training patterns by unique targets
        keys = list(self.patternSet.centers.keys())
        keys.sort()

        # cells in the patterns represent attributes, build attribute array
        attributes = []
        patterns = vectorizeMatrix(self.patternSet.centers[keys[0]])
        for i in range(len(patterns)):
            attributes.append([])
        for key in keys:
            pattern = vectorizeMatrix(self.patternSet.centers[key])
            for i, attribute in enumerate(pattern):
                attributes[i].append(attribute)

        # Now we build as many centers for each attribute as are needed
        # ie. We increase k in k-means until a center is picked with a membership of only 1
        # at which point we pick the previous set of centers for which all memberships are at least 2
        # Note: it would make since that some centers would have only 1 member (outliers), but this
        # produces a sigma of 0.0, which does not allow for fuzziness in the rule's application to untrained targets
        # so we decided a close enough membership was better than none, and that consequent layer updates should account for this after training
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

        #Print Final Rules
        for a, attributeDetails in enumerate(centers):
            for c, center in enumerate(attributeDetails['centers']):
                memString = ', '.join(str(round(x, 3)) for x in attributeDetails['members'][c])
                print("\nC:" + str(a) + ":" + str(c) + "[" + str(round(center, 2)) + "{" + str(round(attributeDetails['sigmas'][c], 2)) + "}]  (" + memString + ")")

        # build the rulesets filling them out with rules corresponding to the centers assembled above
        # form predicate logic by linking up the product nodes to their corresponding rulesets' rules
        # ie. "0" product linked to its coorisponding rule within ruleset 1, 2, ..., n
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
    "Layers link together from 1st to last, and can feedforward in linked list style"
    # Some layers contain just an array of neurons, some contain an array of rulesets in turn containing arrays of rules
    # some contain an array of neurons and a consequence matrix
    # Ultimately, the layer surves the purpose of passing input to output within itself
    # and facilitates the passing of outputs from itself to another layer
    # Specific implementations of the above functionality depends upon the layer's type
    def __init__(self, layerType, prevLayer, neuronCount):
        self.layerType = layerType
        self.prev = prevLayer
        if prevLayer != None:
            prevLayer.next = self
        self.next = None
        self.neurons = []
        self.ruleSets = []
        self.consequences = []
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
    # ie. a specific rule's output from within each of the layer's multiple ruleset
    def getRuleLayerOutputs(self, inputVector):
        outputs = []
        for rIndex, ruleSet in enumerate(self.ruleSets):
            #print("IV:" + str(inputVector[rIndex]) + " RS:" + str(len(ruleSet.rules)))
            if inputVector[rIndex] > len(ruleSet.rules):
                raise NameError('Rule Index in InputVector does not match the number of Rules in this Ruleset!')
            outputs.append(ruleSet.rules[inputVector[rIndex]].output)
        return outputs

    # Consequent training facilitates the networks ability to learn how input from the product/norm layer relate
    # to the target output, in order to produce more accurate output given future product/norm layer inputs
    def adjustConsequent(self, targets):
        if len(targets) != len(self.neurons):
            raise NameError('Output dimension of network does not match that of target!')
        for i, neuron in enumerate(self.neurons):
            error = abs(targets[i] - neuron.output)
            for j in range(len(self.consequences[i])):
                self.consequences[i][j] = self.consequences[i][j] + (eta * ((targets[i] - neuron.output)/len(self.consequences[i])))
            # print("C:" + str(i) + "[" + ", ".join(str(x) for x in self.consequences[i]) + "]")

    # Each Layer has a link to the next layer in order.  Input values are translated from
    # input to output in keeping with the Layer's function
    def feedForward(self):
        if self.layerType == NetLayerType.Input:
            # Input Layer feeds all input to output with no work done
            for neuron in self.neurons:
                neuron.output = neuron.input
        elif self.layerType == NetLayerType.Rules:
            # Each input goes to a specific ruleset
            # The input fed to a particular ruleset is fed in turn to all rules within that ruleset
            # Inputs euclidian distance from the rules centers is passed through
            # a radial basis function (membership) producing the rule's output
            prevOutputs = self.prev.getOutputs()
            for rsIndex, ruleSet in enumerate(self.ruleSets):
                for rIndex, rule in enumerate(ruleSet.rules):
                    # ANFIS on the Euclidian Norm of input to rule center
                    rule.input = prevOutputs[rsIndex]
                    rule.input = euclidianDistance(prevOutputs[rsIndex], rule.center);
                    rule.output = radialBasisFunction(rule.input, rule.sigma)
        elif self.layerType == NetLayerType.ProdNorm:
            # Each product neuron is linked to specific rules within the rulesets of the Rule layer
            # on rule for each ruleset, the product of these inputs is taken and assigned to the neuron's input
            rollingSum = 0.0
            for neuron in self.neurons:
                prevOutputs = self.prev.getRuleLayerOutputs(neuron.inputVector)
                neuron.input = prevOutputs[0]
                for outputValue in prevOutputs[1:]:
                    neuron.input = neuron.input*outputValue
                rollingSum = rollingSum + neuron.input
            # The rolling sum taken during the product step is used to normalize our inputs
            # producing this layer's output value
            if abs(rollingSum) > 0.0:
                for neuron in self.neurons:
                    neuron.output = neuron.input/rollingSum
        elif self.layerType == NetLayerType.Consequent:
            # Consequent Logic takes the networks original inputs and gains them against the
            # consequent matrix, this multiplied with the normalized product layer's output is our final output
            prevOutputs = self.prev.getOutputs()
            layer = self.prev
            while True:
                layer = layer.prev
                if layer.layerType == NetLayerType.Input:
                    break
            netInputs = layer.getOutputs()

            # For each neuron i take Normalized Product Output i * (pi*x + qi*y... + ri)
            for i, neuron in enumerate(self.neurons):
                consequenceSum = 0.0
                for j, inVal in enumerate(netInputs):
                    consequenceSum += inVal*self.consequences[i][j]
                consequenceSum += self.consequences[i][-1]
                neuron.output = prevOutputs[i]*consequenceSum

        # If there is a subsequent layer, feed forward
        if self.next:
            self.next.feedForward()

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
    "RuleSets contain rules which are really just neurons... don't tell anyone though"
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
    trainPercentage = 0.8
    #p = PatternSet('data/optdigits/optdigits-orig.json', trainPercentage)   # 32x32
    #p = PatternSet('data/letter/letter-recognition.json', trainPercentage)  # 1x16 # Try 1 center per attribute, and allow outputs to combine them
    p = PatternSet('data/pendigits/pendigits.json', trainPercentage)        # 1x16 # same as above
    #p = PatternSet('data/semeion/semeion.json', trainPercentage)            # 16x16 # Training set is very limited
    #p = PatternSet('data/semeion/semeion.json', trainPercentage)           # 1593 @ 16x16 # Training set is very limited
    #p = PatternSet('data/optdigits/optdigits.json', trainPercentage)        # 8x8
    
    n = Net(p)
    n.run(PatternType.Train, 0, int(p.count*trainPercentage))
    n.run(PatternType.Test, int(p.count*trainPercentage), p.count)

    p.printConfusionMatrix()
    print("Done")
