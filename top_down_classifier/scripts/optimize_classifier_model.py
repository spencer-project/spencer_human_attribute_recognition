#!/usr/bin/python

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
Loads the specified learned .yaml classifier model (including all volumes of all possible tessellations),
removes volumes which are not used by the learned weak classifiers (thereby significantly reducing feature vector size),
and re-maps the variable indices accordingly. Output is another, more compact and better-performing .yaml file.
NOTE: This script requires at least 8 GB of RAM + swap file (16 GB RAM recommended).
"""

import yaml, sys, re

#
# Parse arguments
#

if len(sys.argv) < 2:
    sys.stderr.write("Missing arguments. Command-line syntax: optimize_classifier_model.py INPUT_FILE.yaml\n")
    sys.exit(1)

inputFilename = sys.argv[1]
print "Opening " + inputFilename + "..."

yamlString = ""
with open(inputFilename) as f:
    for line in f:
        if not line.startswith("%"):
            yamlString += line

#
# Read input YAML file
#

print "Parsing YAML... this may take a while and consume a lot of RAM!"
print "If the process gets killed, buy some more memory or increase swap file size!"

# Some preprocessing necessary to make PyYAML happy
yamlString = re.sub(":", ": ", yamlString)

class CvBoostTree(yaml.YAMLObject):
    yaml_tag = u'tag:yaml.org,2002:opencv-ml-boost-tree'

class CvMatrix(yaml.YAMLObject):
    yaml_tag = u'tag:yaml.org,2002:opencv-matrix'

yamlContent = yaml.load(yamlString)
yamlString = None # free memory


#
# Optimize classifier
#

classifier = yamlContent["classifier"]
print "Classifier consists of %d weak classifiers!" % classifier.ntrees
print "Iterating tree nodes to determine indices of active variables..."

# Iterate over all trees to determine how many variables we need
splitQualities = dict()
originalActiveVars = set()
for tree in classifier.trees:
    for node in tree["nodes"]:
        if "splits" in node:
            for treeSplit in node["splits"]:
                if "var" in treeSplit:
                    oldVarIndex = int(treeSplit["var"])
                    originalActiveVars.add(oldVarIndex)

                    if oldVarIndex in splitQualities:
                        # IMPORTANT: Variables can be used multiple times (by different decision stumps). Store only maximum quality per variable
                        splitQualities[oldVarIndex] = max(float(treeSplit["quality"]), splitQualities[oldVarIndex])
                    else:
                        splitQualities[oldVarIndex] = float(treeSplit["quality"])

originalActiveVars = sorted(list(originalActiveVars))
uniqueVarCount = len(originalActiveVars)
print "Found %d unique active variables!" % uniqueVarCount


#
# Process feature vector entires
#

print "Processing feature vector entries to find which tessellation volumes are required..."
oldFeatureVectorEntries = yamlContent["feature_vector_entries"]

requiredVolumeIndices = set()
requiredFeatureVectorEntries = [ (oldVarIndex, oldFeatureVectorEntries[oldVarIndex]) for oldVarIndex in originalActiveVars ]

# Determine required tessellation volumes
for oldVarIndex, oldFeatureVectorEntry in requiredFeatureVectorEntries:
    requiredVolumeIndices.add( int(oldFeatureVectorEntry["overallVolumeIndex"]) )

# Remove unnecessary tessellation volumes
requiredVolumeIndices = sorted(list(requiredVolumeIndices))
tessellationVolumes = yamlContent["tessellation_volumes"]
tessellationVolumes = [ tessellationVolumes[i] for i in requiredVolumeIndices ]
yamlContent["tessellation_volumes"] = tessellationVolumes

# Remap required tessellation volumes
volumeRemapping = dict()
for i in xrange(len(requiredVolumeIndices)):
    oldVolumeIndex = requiredVolumeIndices[i]
    volumeRemapping[oldVolumeIndex] = i

uniqueVolumeCount = len(requiredVolumeIndices)

# Determine which particular features are required per volume
activeFeaturesPerVolume = [set() for i in xrange(uniqueVolumeCount)]
newVarCounter = 0

print "Counting number of active features for each individual volume..."
for oldVarIndex, oldFeatureVectorEntry in requiredFeatureVectorEntries:
    oldVolumeIndex = int(oldFeatureVectorEntry["overallVolumeIndex"])
    newVolumeIndex = volumeRemapping[oldVolumeIndex]
    featureIndex = int(oldFeatureVectorEntry["featureIndex"])
    activeFeaturesPerVolume[newVolumeIndex].add(featureIndex)

# FIXME: Put this into list of tessellation_volumes    
activeFeaturesPerVolume = [ sorted(list(activeFeatures)) for activeFeatures in activeFeaturesPerVolume ]
yamlContent["active_features_per_volume"] = activeFeaturesPerVolume

# Generate lookup table needed to generate new feature mapping
accumulatedNumberOfVarsBeforeVolume = []
accumulatedNumberOfVars = 0
volumeCounter = 0
for activeFeatures in activeFeaturesPerVolume:
    accumulatedNumberOfVarsBeforeVolume.append(accumulatedNumberOfVars)
    accumulatedNumberOfVars += len(activeFeaturesPerVolume[volumeCounter])
    volumeCounter += 1

# Finally remap variable indices
# +Generate new feature vector entries
varRemapping = dict()
newFeatureVectorLength = accumulatedNumberOfVars
newFeatureVectorEntries = [ dict() for i in xrange(newFeatureVectorLength)]

print "Number of required feature vector entries: %d --> expected feature vector length %d" % (len(requiredFeatureVectorEntries), newFeatureVectorLength)
print "Remapping variable indices..."
for oldVarIndex, oldFeatureVectorEntry in requiredFeatureVectorEntries:
    # Remap volume index
    oldVolumeIndex = int(oldFeatureVectorEntry["overallVolumeIndex"])
    newVolumeIndex = volumeRemapping[oldVolumeIndex]
    # Remap feature index
    oldFeatureIndex = int(oldFeatureVectorEntry["featureIndex"])
    newFeatureIndex = activeFeaturesPerVolume[newVolumeIndex].index(oldFeatureIndex)
    # Remap variable index
    newVarIndex = accumulatedNumberOfVarsBeforeVolume[newVolumeIndex] + newFeatureIndex
    varRemapping[oldVarIndex] = newVarIndex
    # Update feature vector entries
    newFeatureVectorEntries[newVarIndex]["overallVolumeIndex"] = newVolumeIndex
    newFeatureVectorEntries[newVarIndex]["featureIndex"] = newFeatureIndex
    # Store quality of corresponding decision tree split to be able to sort volumes or features by quality
    newFeatureVectorEntries[newVarIndex]["quality"] = splitQualities[oldVarIndex]

yamlContent["feature_vector_entries"] = newFeatureVectorEntries


#
# Apply remapping to classifier
#
print "Applying variable index remapping to classifier..."
for tree in classifier.trees:
    for node in tree["nodes"]:
        if "splits" in node:
            for treeSplit in node["splits"]:
                if "var" in treeSplit:
                    oldVarIndex = int(treeSplit["var"])
                    treeSplit["var"] = varRemapping[oldVarIndex]


# Set classifier.var_all, var_count, ord_var_count to new number of variables
classifier.var_all = classifier.var_count = classifier.ord_var_count = newFeatureVectorLength

# Update size of var_type array
classifier.var_type = classifier.var_type[0:newFeatureVectorLength]

# Update general meta-info
print "Updating general meta-info..."
assert(yamlContent["num_overall_volumes"] != 0) # make sure field exists
yamlContent["num_overall_volumes"] = uniqueVolumeCount

oldFeatureVectorLength = yamlContent["feature_vector_length"]
yamlContent["feature_vector_length"] = newFeatureVectorLength

del yamlContent["num_tessellations"]
yamlContent["optimized"] = True


print "Resulting feature vector has length %d (was %d), size reduced to %.1f%% of original size!" % (newFeatureVectorLength, oldFeatureVectorLength, 100.0 * newFeatureVectorLength / oldFeatureVectorLength )



#
# Write output YAML file
#

outputFilename = inputFilename.replace(".yaml", "_optimized.yaml")
outputFilename = outputFilename.replace(".adaboost", "_optimized.adaboost")  # old file ending
assert(outputFilename != inputFilename)

# Makes indentation compatible with OpenCV parser
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

yamlString = "%YAML:1.0\n" + yaml.dump(yamlContent, Dumper=MyDumper, default_flow_style=False)

print "Writing YAML..."
with open(outputFilename, "w") as f:
    f.write(yamlString)
    
print "Done!"



