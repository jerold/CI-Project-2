#!/usr/bin/python
# ANFIS Network

import sys
import random
import math
import time
import patternSet
from patternSet import PatternSet

eta = 1.00

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

# Combination of two vectors
def linearCombination(p, q):
    lSum = 0.0
    for i in range(len(p)):
        lSum = lSum + p[i]*q[i]
    return lSum

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

# turns a matrix into a vector
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
                self.layers[NetLayerType.Consequent].adjustWeights(self.patternSet.targetVector(patterns[i]['t']))
                #self.layers[NetLayerType.Consequent].adjustConsequent(self.patternSet.targetVector(patterns[i]['t']))
            else:
                # OPTION 1 --WINNER--
                #self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[NetLayerType.Consequent].getOutputs())

                # OPTION 2
                self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[NetLayerType.ProdNorm].getOutputs())

                # print("\nOutputs")
                # print("[" + ", ".join(str(int(x+0.2)) for x in self.layers[NetLayerType.ProdNorm].getOutputs()) + "]")
                # print("Target")
                # print(self.patternSet.targetVector(patterns[i]['t']))

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
        attributesSigmas = []
        patterns = vectorizeMatrix(self.patternSet.centers[keys[0]])
        for i in range(len(patterns)):
            attributes.append([])
            attributesSigmas.append([])
        for key in keys:
            pattern = vectorizeMatrix(self.patternSet.centers[key])
            sigma = vectorizeMatrix(self.patternSet.sigmas[key])
            for i, attribute in enumerate(pattern):
                attributes[i].append(attribute)
                attributesSigmas[i].append(sigma[i])

        # OPTION 1 K-MEANS RULESET CREATION
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

        # OPTION 2 RULE PER TARGET PER ATTRIBUTE --WINNER--
        # Every RuleSet will have a rule for each target
        # the rule is built from the training pattern set's center and sigma for that target
        # centers = []
        # for k, attribute in enumerate(attributes):
        #     attributeCenters = {'centers':[], 'members':[], 'sigmas':[]}
        #     # print(k)
        #     for j, jthOutputRule in enumerate(attribute):
        #         # print(str(j) + " " + str(jthOutputRule) + " " + str(attributesSigmas[k][j]))
        #         attributeCenters['centers'].append(jthOutputRule)
        #         attributeCenters['members'].append([j])
        #         attributeCenters['sigmas'].append(attributesSigmas[k][j])
        #     centers.append(attributeCenters)

        #Print Final Rules
        # for a, attributeDetails in enumerate(centers):
        #     for c, center in enumerate(attributeDetails['centers']):
        #         memString = ', '.join(str(round(x, 3)) for x in attributeDetails['members'][c])
        #         print("C:" + str(a) + ":" + str(c) + "[" + str(round(center, 2)) + "{" + str(round(attributeDetails['sigmas'][c], 2)) + "}]  (" + memString + ")")

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

            # OPTION 1 --WINNER--
            # We backprop only one time per neuron
            for j in range(len(prevOutputs)):
                neuron.weightDeltas[j] = eta * (targets[k] - neuron.output) * prevOutputs[j]
                neuron.weights[j] = neuron.weights[j] + neuron.weightDeltas[j]
                if neuron.weights[j] > 9999999:
                    raise NameError('Divergent Weights!')

            # OPTION 2
            # We backprop until the weights perfectly account for target-output difference
            # deltaSum = 1
            # while deltaSum > 0.1:
            #     deltaSum = 0.0
            #     for j in range(len(prevOutputs)):
            #         deltaSum = deltaSum + (targets[k] - neuron.output)
            #         neuron.weightDeltas[j] = eta * (targets[k] - neuron.output) * prevOutputs[j]
            #         neuron.weights[j] = neuron.weights[j] + neuron.weightDeltas[j]
            #         if neuron.weights[j] > 9999999:
            #             raise NameError('Divergent Weights!')
            #     neuron.output = linearCombination(prevOutputs, neuron.weights)


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
        # elif self.layerType == NetLayerType.Consequent:
        #     # Consequent Logic takes the networks original inputs and gains them against the
        #     # consequent matrix, this multiplied with the normalized product layer's output is our final output
        #     prevOutputs = self.prev.getOutputs()
        #     layer = self.prev
        #     while True:
        #         layer = layer.prev
        #         if layer.layerType == NetLayerType.Input:
        #             break
        #     netInputs = layer.getOutputs()

        #     # For each neuron i take Normalized Product Output i * (pi*x + qi*y... + ri)
        #     for i, neuron in enumerate(self.neurons):
        #         consequenceSum = 0.0
        #         for j, inVal in enumerate(netInputs):
        #             consequenceSum += inVal*self.consequences[i][j]
        #         consequenceSum += self.consequences[i][-1]
        #         neuron.output = prevOutputs[i]*consequenceSum

        elif self.layerType == NetLayerType.Consequent:
            # Linear Combination of Hidden layer outputs and associated weights
            for neuron in self.neurons:
                prevOutputs = self.prev.getOutputs()
                if len(neuron.weights) != len(prevOutputs):
                    raise NameError('Weights dimension does not match that of previous Layer outputs!')
                neuron.output = linearCombination(prevOutputs, neuron.weights)


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
    #p = PatternSet('data/letter/letter-recognition.json', trainPercentage)  # 20000 @ 1x16 # Try 1 center per attribute, and allow outputs to combine them
    #p = PatternSet('data/pendigits/pendigits.json', trainPercentage)        # 10992 @ 1x16 # same as above
    #p = PatternSet('data/semeion/semeion.json', trainPercentage)            # 1593 @ 16x16 # Training set is very limited
    p = PatternSet('data/optdigits/optdigits.json', trainPercentage)        # 5620 @ 8x8
    
    n = Net(p)
    n.run(PatternType.Train, 0, int(p.count*trainPercentage))
    n.run(PatternType.Test, int(p.count*trainPercentage), p.count)

    p.printConfusionMatrix()
    print("Done")
