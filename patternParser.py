#!/usr/bin/python

import json
import itertools

import math
import random

optdigits = {'inFiles':['data/optdigits/optdigits-orig.tra',
                        'data/optdigits/optdigits-orig.windep',
                        'data/optdigits/optdigits-orig.wdep'],
             'outFile':'data/optdigits/optdigits.json',
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
             'outFile':'data/semeion/semeion.json',
             'width':16,
             'height':16}


def parseOptdigits(lines, w, h):
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
            # var = raw_input("Cont?" + str(hi))
            pattern = []
            hi = 0
        else:
            # Bad line
            print('Bad Line: [' + str(''.join(line)) + ']')
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
            #var = raw_input("cont")
            patSet.append({'p':pattern, 't':patternTarget})
        else:
            var = raw_input("Bad line [" + line + "]")
    return patSet

def mygrouper(n, iterable):
    "http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n"
    args = [iter(iterable)] * n
    return ([e for e in t if e != None] for t in itertools.zip_longest(*args))

def buildKMeansCenters(patterns, w, h, k):
    centers = {}
    for i in range(k):
        centers[i] = emptyPattern(w, h)
        if h > 1:
            centers[i][random.randint(0,h-1)][random.randint(0,w-1)] = 1
        else:
            centers[i][random.randint(w)] = 1
    dist = 100
    while dist > 3:
        tempCenters = adjustCenters(patterns, centers)
        dist = 0
        for i in range(k):
            dist = dist + euclidianDistance(centers[i], tempCenters[i])
        centers = tempCenters
        print(dist)
    for i in range(k):
        if h > 1:
            for j in range(len(centers[i])):
                print(''.join(str(x+0.5).split('.')[0] for x in centers[i][j]))
        else:
            print(', '.join(str(x).split('.')[0] for x in centers[i]))
        print(k)
        var = raw_input("cont?")

def adjustCenters(patterns, centers):
    groups = {}
    for i in range(len(centers)):
        groups[i] = []
    for pattern in patterns:
        bestDist = 99999
        bestPat = 0
        pat = 0
        for key in centers.keys():
            center = centers[key]
            dist = euclidianDistance(pattern['p'], center)
            if dist < bestDist:
                bestDist = dist
                bestPat = pat
            pat = pat + 1
        groups[bestPat].append(pattern)
    newCenters = {}
    for i in range(len(centers)):
        if len(groups[i]) > 0:
            newCenters[i] = buildMeanPattern(groups[i])
        else:
            newCenters[i] = centers[i]
    return newCenters

def euclidianDistance(p, q):
    sumOfSquares = 0.0
    if isinstance(p[0], list):
        for i in range(len(p)):
            for j in range(len(p[i])):
                sumOfSquares = sumOfSquares + ((p[i][j]-q[i][j])*(p[i][j]-q[i][j]))
    else:
        for i in range(len(p)):
                sumOfSquares = sumOfSquares + ((p[i][j]-q[i][j])*(p[i][j]-q[i][j]))
    return math.sqrt(sumOfSquares)


def buildCenters(patterns, w, h):
    centersTargets = {}
    for pattern in patterns:
        if pattern['t'] not in centersTargets:
            centersTargets[pattern['t']] = []
        centersTargets[pattern['t']].append(pattern)
    centers = {}
    # build center as mean of all trained k patterns, and sigma as standard deviation
    for k in centersTargets.keys():
        kPats = centersTargets[k]
        meanPattern = buildMeanPattern(kPats)

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
                print(mPat[j])
                print(pat['p'][j])
                mPat[j] = mPat[j] + pat['p'][j]
    if h > 1:
        for i in range(h):
            for j in range(w):
                mPat[i][j] = mPat[i][j] / len(patterns)
    else:
        for j in range(w):
            mPat[j] = mPat[j] / len(patterns)
    # if h > 1:
    #     for i in range(len(mPat)):
    #         print(''.join(str(x+0.5).split('.')[0] for x in mPat[i]))
    # else:
    #     print(', '.join(str(x).split('.')[0] for x in mPat))
    # var = raw_input("cont?")
    return mPat

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

if __name__=="__main__":
    parseSet = optdigits
    #parseSet = letterRecognition
    #parseSet = pendigits
    #parseSet = semeion
    lines = []
    for fileName in parseSet['inFiles']:
        with open(fileName) as file:
            fileLines = file.readlines()
            for line in fileLines:
                lines.append(line)
    patternSet = parseOptdigits(lines, parseSet['width'], parseSet['height'])
    #patternSet = parseLetterRecognition(lines)
    #patternSet = parsePendigits(lines)
    #patternSet = parseSemeion(lines, parseSet['width'], parseSet['height'])
    
    buildKMeansCenters(patternSet, parseSet['width'], parseSet['height'], 10)
    #buildCenters(patternSet, parseSet['width'], parseSet['height'])
    print("pats: " + str(len(patternSet)))
    with open(parseSet['outFile'], 'w+') as outfile:
        data = {'count':len(patternSet),
                'width':parseSet['width'],
                'height':parseSet['height'],
                'patterns':patternSet}
        json.dump(data, outfile)
