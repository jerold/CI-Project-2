#!/usr/bin/python

import json

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
            pattern.append(line)
            hi = hi + 1
        elif hi == h:
            # End of Pattern and Target Line
            #patternTarget = int(''.join(line))
            patternTarget = ''.join(line)
            patSet.append({'p':pattern, 't':patternTarget})
            # print('Target:' + str(''.join(line)))
            # var = input("Cont?" + str(hi))
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
        patternTarget = line[0]
        patSet.append({'p':pattern, 't':patternTarget})
    return patSet

if __name__=="__main__":
    #parseSet = optdigits
    parseSet = letterRecognition
    lines = []
    for fileName in parseSet['inFiles']:
        with open(fileName) as file:
            fileLines = file.readlines()
            for line in fileLines:
                lines.append(line)
    #patternSet = parseOptdigits(lines, parseSet['width'], parseSet['height'])
    patternSet = parseLetterRecognition(lines)
    print("pats: " + str(len(patternSet)))
    with open(parseSet['outFile'], 'w+') as outfile:
        data = {'count':len(patternSet),
                'width':parseSet['width'],
                'height':parseSet['height'],
                'patterns':patternSet}
        json.dump(data, outfile)
