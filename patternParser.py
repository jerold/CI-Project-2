#!/usr/bin/python

import json
import itertools

import math
import random

optdigits = {'inFiles':['data/optdigits/optdigits.tra',
                        'data/optdigits/optdigits.tes'],
             'outFile':'data/optdigits/optdigits.json',
             'width':8,
             'height':8}

optdigitsOrig = {'inFiles':['data/optdigits/optdigits-orig.tra',
                        'data/optdigits/optdigits-orig.windep',
                        'data/optdigits/optdigits-orig.wdep'],
             'outFile':'data/optdigits/optdigits-orig.json',
             'width':32,
             'height':32}

letterRecognition = {'inFiles':['data/letter/letter-recognition.data'],
             'outFile':'data/letter/letter-recognition.json',
             'width':16,
             'height':1}

pendigits = {'inFiles':['data/pendigits/pendigits.tra',
                        'data/pendigits/pendigits.tes'],
             'outFile':'data/pendigits/pendigits.json',
             'width':16,
             'height':1}

semeion = {'inFiles':['data/semeion/semeion.data'],
             'outFile':'data/semeion/semeionTT.json',
             'width':16,
             'height':16}

def parseOptdigits(lines, w, h):
    patSet = []
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        pattern = list(mygrouper(w, line[:len(line)]))
        patternTarget = pattern[h][0]
        patternTarget = int(patternTarget)
        pattern = pattern[:h]
        for i in range(len(pattern)):
            for j in range(len(pattern[i])):
                pattern[i][j] = int(pattern[i][j])
        patSet.append({'p':pattern, 't':patternTarget})
    return patSet

def parseOptdigitsOrig(lines, w, h):
    patSet = []
    pattern = []
    patternTarget = 0
    hi = 0
    for line in lines:
        line = list(line)
        # Cuts off newline \n
        line = line[:len(line)-1]
        if len(line) == w:
            # Pattern Line
            for i in range(len(line)):
                line[i] = int(line[i])
            pattern.append(line)
            hi = hi + 1
        elif hi == h:
            # End of Pattern and Target Line
            patternTarget = int(''.join(line))
            #patternTarget = ''.join(line)
            patSet.append({'p':pattern, 't':patternTarget})
            # print('Target:' + str(''.join(line)))
            # var = input("Cont?" + str(hi))
            pattern = []
            hi = 0
        else:
            # Bad line
            #print('Bad Line: [' + str(''.join(line)) + ']')
            pattern = []
            hi = 0
    return patSet

def parseLetterRecognition(lines):
    patSet = []
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        pattern = line[1:]
        for i in range(len(pattern)):
            pattern[i] = int(pattern[i])
        patternTarget = line[0]
        patSet.append({'p':pattern, 't':patternTarget})
    return patSet

def parsePendigits(lines):
    patSet = []
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        pattern = line[:len(line)-1]
        for i in range(len(pattern)):
            pattern[i] = int(pattern[i])
        patternTarget = int(line[len(line)-1])
        patSet.append({'p':pattern, 't':patternTarget})
    return patSet

def parseSemeion(lines, w, h):
    patSet = []
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(' ')
        pattern = list(mygrouper(w, line[:len(line)-1]))
        patternTarget = pattern[h]
        pattern = pattern[:h]
        if len(patternTarget) == 10:
            for i in range(len(pattern)):
                for j in range(len(pattern[i])):
                    pattern[i][j] = int(pattern[i][j].split('.')[0])
            for i in range(len(patternTarget)):
                patternTarget[i] = int(patternTarget[i])
            #print(patternTarget)
            patternTarget = list(itertools.compress([0,1,2,3,4,5,6,7,8,9], patternTarget))[0]
            #print(line)
            #for i in range(len(pattern)):
            #    print(''.join(str(x) for x in pattern[i]))
            #print(patternTarget)
            #var = input("cont")
            patSet.append({'p':pattern, 't':patternTarget})
        else:
            var = input("Bad line [" + line + "]")
    random.shuffle(patSet)
    return patSet

def mygrouper(n, iterable):
    "http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n"
    args = [iter(iterable)] * n
    return ([e for e in t if e != None] for t in itertools.izip_longest(*args))

def buildKMeansCenters(patterns, w, h, k):
    centers = {}
    for i in range(k):
        centers[i] = emptyPattern(w, h)
        if h > 1:
            centers[i][random.randint(0,h-1)][random.randint(0,w-1)] = 1
        else:
            centers[i][random.randint(0,w-1)] = 1
    dist = 100
    while dist > 3:
        tempCenters = adjustCenters(patterns, centers)
        dist = 0
        for i in range(k):
            dist = dist + euclidianDistance(centers[i], tempCenters[i])
        centers = tempCenters
        print(dist)
    for i in range(k):
        printPattern(centers[i])
        print(i)
    return centers

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

def euclidianDistance(p, q):
    sumOfSquares = 0.0
    if isinstance(p[0], list):
        for i in range(len(p)):
            for j in range(len(p[i])):
                sumOfSquares = sumOfSquares + ((p[i][j]-q[i][j])*(p[i][j]-q[i][j]))
    else:
        for i in range(len(p)):
            sumOfSquares = sumOfSquares + ((p[i]-q[i])*(p[i]-q[i]))
    return math.sqrt(sumOfSquares)


def buildCentersAndSigmas(patterns):
    centersTargets = {}
    for pattern in patterns:
        if pattern['t'] not in centersTargets:
            centersTargets[pattern['t']] = []
        centersTargets[pattern['t']].append(pattern)
    centers = {}
    sigmas = {}
    print("Found " + str(len(centersTargets)) + " targets.")
    # build center as mean of all trained k patterns, and sigma as standard deviation
    for k in centersTargets.keys():
        kPats = centersTargets[k]
        centers[k] = buildMeanPattern(kPats)

    print("Centers Post Average:")
    for k in centersTargets.keys():
        print(k)
        printPattern(centers[k])

    # Build Sigmas for each space
    print("Sigma:")
    for k in centersTargets.keys():
        sigmas[k] = buildSigmaPattern(centers[k], kPats)
        printPattern(sigmas[k])

    # refine centers using k-means
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
        print("dist:" + str(dist) + ", delta:" + str(distDelta))

    print("Centers Post K-means:")
    for k in centersTargets.keys():
        print(k)
        printPattern(centers[k])

    return {'centers':centers, 'sigmas':sigmas}

def buildMeanPattern(patterns):
    h = 0
    w = len(patterns[0]['p'])
    if isinstance(patterns[0]['p'][0], list):
        h = len(patterns[0]['p'])
        w = len(patterns[0]['p'][0])
    mPat = emptyPattern(w, h)
    for pat in patterns:
        #print(pat['p'])
        if h > 1:
            #print(str(len(pat['p'])) + " x " + str(len(pat['p'][0])))
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

def printPattern(pattern):
    tolerance = 0.7
    if isinstance(pattern[0], list):
        for i in range(len(pattern)):
            print(', '.join(str(round(x+tolerance,3)) for x in pattern[i]))
    else:
        print(', '.join(str(round(x,3)) for x in pattern))


if __name__=="__main__":
    #parseSet = optdigits
    #parseSet = optdigitsOrig
    #parseSet = letterRecognition
    #parseSet = pendigits
    parseSet = semeion
    lines = []
    for fileName in parseSet['inFiles']:
        with open(fileName) as file:
            fileLines = file.readlines()
            for line in fileLines:
                lines.append(line)
    #patternSet = parseOptdigits(lines, parseSet['width'], parseSet['height'])
    #patternSet = parseOptdigitsOrig(lines, parseSet['width'], parseSet['height'])
    #patternSet = parseLetterRecognition(lines)
    #patternSet = parsePendigits(lines)
    patternSet = parseSemeion(lines, parseSet['width'], parseSet['height'])
    
    # buildKMeansCenters(patternSet, parseSet['width'], parseSet['height'], 20)
    centerSigmas = buildCentersAndSigmas(patternSet[:800])
    centers = centerSigmas['centers']
    sigmas = centerSigmas['sigmas']
    
    print("pats: " + str(len(patternSet)))
    with open(parseSet['outFile'], 'w+') as outfile:
        data = {'count':len(patternSet[800:]),
                'width':parseSet['width'],
                'height':parseSet['height'],
                'patterns':patternSet[800:],
                'centers':centers,
                'sigmas':sigmas}
        json.dump(data, outfile)


