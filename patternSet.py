#!/usr/bin/python
# patternSet.py

import json
import random
import math

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

# Centers are built for each of the targets in two steps
# First an average is built for each target from each pattern of that target type
# Next we run k-means on the new centers against all patterns
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

    # OPTIONAL k-MEANS CENTER SOFTENING --WINNER--
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
        #print("dist:" + str(round(dist, 4)) + ", delta:" + str(round(distDelta, 4)))

    # #Build Sigmas for each space
    for k in centersTargets.keys():
        sigmas[k] = buildSigmaPattern(centers[k], kPats)

    return {'centers':centers, 'sigmas':sigmas}

# given several patterns we create an average pattern of the same dimension
# This method works for 2D arrays, 1D arrays, and scalers
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

# This pattern shows us where the fuzziness is in our average
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

# Used in k-means, here we take an average of the member patterns to construct a new center
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

# Creates and empty pattern of the given dimensionality
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
        matrixSum = 0.0
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
            matrixSum = matrixSum + rowSum
            precision.append(rowSum)
            i += 1
        for i, elem in enumerate(diagonal):
            if abs(precision[i]) > 0.0 and abs(recall[i]) > 0.0:
                print(str(keys[i]) + ", " + str(round(elem / precision[i], 4)) + ", " + str(round(elem/recall[i], 4)))
        print("Overall Correct: " + str(round(sum(diagonal)/matrixSum, 4)))
        
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