#!/usr/bin/env python

# Software License Agreement (BSD License)
#
#  Copyright (c) 2014-2015, Timm Linder, Social Robotics Lab, University of Freiburg
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Creates a text file with image filenames and corresponding labels
for training and test set.
"""

import sys, re, os, random

# Parameters
trainRatio = 0.5
activeSequences = [ 1 , 2, 3, 4 ]
subsampling = 5  # take every n-th frame for sequences != sequence 1
shuffleFramesInterPerson = True
balanceSamples = True
directory = "/home/linder/Datasets/srl_dataset_clouds/"
category="long_trousers" # "gender"

# Initialize random number generator
randomSeed = random.randint(0, 100000)
if len(sys.argv) > 1:
    randomSeed = int(sys.argv[1])
random.seed(randomSeed * 6666)

# Read groundtruth file
labels = dict()
persons = []
labelsFile = "groundtruth.txt"
numPositive = 0
with open(labelsFile, "r") as f:
  for line in f:
    if line.startswith("#"):
        continue

    tokens = line.split()
    personId = int(tokens[0])
    gender = str(tokens[1])
    age = int(tokens[2])
    hasLongHair = int(tokens[10])
    hasGlasses = int(tokens[4])
    hasJacket = int(tokens[5])
    hasHat = int(tokens[6])
    hasLongSleeves = int(tokens[9])
    hasLongTrousers = int(tokens[11])

    if gender == "female":
        genderLabel = 0
    elif gender == "male":
        genderLabel = 1
    else:
        raise Exception("Unknown gender: %s" % gender)

    if category == "gender":
        labels[personId] = genderLabel
    elif category == "long_hair":
        labels[personId] = hasLongHair
    elif category == "glasses":
        labels[personId] = hasGlasses
    elif category == "jacket":
        labels[personId] = hasJacket
    elif category == "hat":
        labels[personId] = hasHat
    elif category == "long_sleeves":
        labels[personId] = hasLongSleeves
    elif category == "long_trousers":
        labels[personId] = hasLongTrousers
    elif category == "random_label_per_person":
        labels[personId] = random.randint(0, 1)
    else:
        raise Exception("Unknown category: %s" % category)

    persons.append(personId)
    numPositive += 1 if labels[personId] else 0

print "Number of persons in positive class: %d (%.1f%%)" % (numPositive, numPositive * 100.0 / len(persons))

# This is to ensure lists are really shuffled randomly
for i in xrange(0, 5 + randomSeed % 9):
    random.shuffle(persons)

# Create folds
pivotIndex = int(len(persons) * trainRatio)
trainingSet = persons[0:pivotIndex]
testSet = persons[pivotIndex:]

numPositiveTraining = numPositiveTest = 0
for personId in trainingSet:
    numPositiveTraining += 1 if labels[personId] else 0
for personId in testSet:
    numPositiveTest += 1 if labels[personId] else 0

numNegativeTraining = len(trainingSet) - numPositiveTraining
numNegativeTest = len(testSet) - numPositiveTest


personFolders = [x[0] for x in os.walk(directory)]
random.shuffle(personFolders)

trainingLines = []
testLines = []

personsWithoutLabel = set()

for personFolder in personFolders:
    print "Examining folder: " + personFolder + "..."
    filenames = next(os.walk(personFolder))[2]

    pattern = re.compile("person_([0-9]+)_([0-9]+)_([0-9]+)_cloud.pcd")
    for filename in filenames:
        match = pattern.match(filename)
        if match:
            personId = int(match.group(1))
            sequenceId = int(match.group(2))
            frameId = int(match.group(3))

            if sequenceId in activeSequences:
                if personId in labels:
                    label = labels[personId]
                    line = "%s/%s %d\n" % (personFolder, match.group(0), label)

                    sample = (line, sequenceId, label)
                    if personId in trainingSet:
                        trainingLines.append( sample )
                    else:
                        testLines.append( sample )
                else:
                    personsWithoutLabel.add(personId)

if personsWithoutLabel:
    sys.stderr.write("\nThe following persons have NO label! Check annotations:\n%s\n\n" % str(list(personsWithoutLabel)))

if shuffleFramesInterPerson:
    random.shuffle(trainingLines)
    random.shuffle(testLines)

def subsample(lines):
    if subsampling > 1 and activeSequences != [ 1 ]:
        desiredLineCount = len(lines) / subsampling
        while len(lines) > desiredLineCount:
            randomIndex = random.randint(0, len(lines)-1)
            line = lines[randomIndex]
            if line[1] != 1: # don't subsample in sequence 1
                del lines[randomIndex]

print "Subsampling (except for sequence 1)..."
subsample(trainingLines)
subsample(testLines)

def getAbsFrequencyPerLabel(lines):
    absFrequencies = dict()
    for line in lines:
        label = line[2]
        if not label in absFrequencies:
            absFrequencies[label] = 0
        absFrequencies[label] += 1
    return absFrequencies

def performSampleBalancing(lines):
    absFrequencies = getAbsFrequencyPerLabel(lines)

    if absFrequencies[0] == absFrequencies[1]:
        return
    elif absFrequencies[0] < absFrequencies[1]:
        classToUndersample = 1
        numSamplesToRemove = absFrequencies[1] - absFrequencies[0]
    else:
        classToUndersample = 0
        numSamplesToRemove = absFrequencies[0] - absFrequencies[1]

    while numSamplesToRemove > 0:
        for i in xrange(0, len(lines)):
            # NOTE: This implementation is not very efficient as we always start back at the beginning of the list
            # once we have deleted an item
            line = lines[i]
            label = line[2]
            if label == classToUndersample:
                del lines[i]
                numSamplesToRemove -= 1
                break

    absFrequencies = getAbsFrequencyPerLabel(lines)
    assert(absFrequencies[0] == absFrequencies[1])

if balanceSamples:
    print "Balancing samples in training set..."
    performSampleBalancing(trainingLines)
    print "Balancing samples in test set..."
    performSampleBalancing(testLines)

metaData = "# Category: %s, Random seed: %d - Active sequences: %s - Shuffle samples inter-person: %s - Subsampling: %d - Balance samples: %s\n" % (category, randomSeed, str(activeSequences), str(shuffleFramesInterPerson), subsampling, str(balanceSamples))

foldFolder = "fold%d" % randomSeed
os.mkdir(foldFolder)

with open(foldFolder + "/train.txt", "w") as f:
    absFrequencies = getAbsFrequencyPerLabel(trainingLines)
    f.write(metaData)
    f.write("# No. individuals: %d - No. samples total: %d - Num positive class samples: %d - Num negative class samples: %d\n" % (len(trainingSet), len(trainingLines), absFrequencies[1], absFrequencies[0] ))
    for line in trainingLines:
        f.write(line[0])

with open(foldFolder + "/val.txt", "w") as f:
    absFrequencies = getAbsFrequencyPerLabel(testLines)
    f.write(metaData)
    f.write("# No. individuals: %d - No. samples total: %d - Num positive class samples: %d - Num negative class samples: %d\n" % (len(testSet), len(testLines), absFrequencies[1], absFrequencies[0] ))
    for line in testLines:
        f.write(line[0])

print "Active sequences: %s, category: %s, random seed: %d, shuffle samples inter-person: %s, subsampling: %d, balance samples: %s" % (str(activeSequences), category, randomSeed, str(shuffleFramesInterPerson), subsampling, str(balanceSamples))
print "Results have been written to train.txt and val.txt!"
print "Training set consists of %d persons (%d samples), test set of %d persons (%d samples)!" % (len(trainingSet), len(trainingLines), len(testSet), len(testLines))
