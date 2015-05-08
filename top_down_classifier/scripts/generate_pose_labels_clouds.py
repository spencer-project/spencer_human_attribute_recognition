#!/usr/bin/env python

# Creates a text file with image filenames and corresponding labels
# for training and test set.
#

import sys, re, os, random

# Parameters
trainRatio = 0.5
activeSequences = [ 1, 2, 3, 4 ]
subsampling = 5  # take every n-th frame for sequences != sequence 1
shuffleFramesInterPerson = True
directory = "/home/linder/Datasets/srl_dataset_clouds/"


# Read groundtruth file
labels = dict()
persons = []
labelsFile = "groundtruth.txt"
with open(labelsFile, "r") as f:
  for line in f:
    tokens = line.split('\t')
    personId = int(tokens[0])
    gender = str(tokens[1])
    age = int(tokens[2])

    if gender == "female":
        label = 0
    elif gender == "male":
        label = 1
    else:
        raise Exception("Unknown gender: %s" % gender)

    labels[personId] = label
    persons.append(personId)

randomSeed = random.randint(0, 100000)
if len(sys.argv) > 1:
    randomSeed = int(sys.argv[1])

# This is to ensure lists are really shuffled randomly
random.seed(randomSeed * 6666)
for i in xrange(0, 5 + randomSeed % 9):
    random.shuffle(persons)


# Create folds
pivotIndex = int(len(persons) * trainRatio)
trainingSet = persons[0:pivotIndex]
testSet = persons[pivotIndex:]

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

                    if personId in trainingSet:
                        trainingLines.append( (line, sequenceId) )
                    else:
                        testLines.append( (line, sequenceId) )
                else:
                    personsWithoutLabel.add(personId)

if personsWithoutLabel:
    sys.stderr.write("\nThe following persons have NO label! Check annotations:\n%s\n\n" % str(list(personsWithoutLabel)))

if shuffleFramesInterPerson:
    random.shuffle(trainingLines)
    random.shuffle(testLines)

def subsample(lines):
    if subsampling > 1:
        desiredLineCount = len(lines) / subsampling
        while len(lines) > desiredLineCount:
            randomIndex = random.randint(0, len(lines)-1)
            line = lines[randomIndex]
            if line[1] != 1: # don't subsample in sequence 1
                del lines[randomIndex]


print "Subsampling (except for sequence 1)..."
subsample(trainingLines)
subsample(testLines)

metaData = "# Random seed: %d - Active sequences: %s - Shuffle frames inter-person: %s - Subsampling: %d\n" % (randomSeed, str(activeSequences), str(shuffleFramesInterPerson), subsampling)

with open("train.txt", "w") as f:
    f.write(metaData)
    f.write("# No. individuals: %d - No. frames total: %d\n" % (len(trainingSet), len(trainingLines)) )
    for line in trainingLines:
        f.write(line[0])

with open("val.txt", "w") as f:
    f.write(metaData)
    f.write("# No. individuals: %d - No. frames total: %d\n" % (len(testSet), len(testLines)) )
    for line in testLines:
        f.write(line[0])
 
print "Active sequences: %s, random seed: %d, shuffle frames inter-person: %s, subsampling: %d" % (str(activeSequences), randomSeed, str(shuffleFramesInterPerson), subsampling)
print "Results have been written to train.txt and val.txt!"
print "Training set consists of %d persons (%d frames), test set of %d persons (%d frames)!" % (len(trainingSet), len(trainingLines), len(testSet), len(testLines))