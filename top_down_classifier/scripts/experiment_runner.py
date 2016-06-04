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
Script for batch-running multiple experiments in a sequence.
(c) 2014-2015 Timm Linder, Social Robotics Lab, University of Freiburg
"""

import rospkg, os, sys, subprocess, copy, math, time, random

class Args(dict):
    def vary(self, argName, argValue):
        c = copy.deepcopy(self)
        c[argName] = argValue
        return c

##################################################################################################################
# ADD/MODIFY EXPERIMENTS HERE. WHEN ADDING A NEW EXPERIMENT, INCREMENT THE ID PLEASE
# TO MAINTAIN ORDER IN THE DIRECTORY STRUCTURE.
##################################################################################################################

# Modifications done here will affect all experiments
baseArgs = Args()
baseArgs["fold"] = 2000
baseArgs["num_folds"] = 10
baseArgs["show_tessellations"] = False
baseArgs["weak_count"] = 500
baseArgs["parent_volume_height"] = 1.8
baseArgs["parent_volume_z_offset"] = 0.0
baseArgs["scale_z_to"] = -1.0
baseArgs["_name"] = "train_top_down_%d" % random.randint(0, 10000)  # random ID to allow multiple parallel executions

experiments = dict()

################################################################################################
# NOTE: Need to recompile with specified set of features before running these experiments.
# Compile flags: no USE_EXTENT_FEATURES, no color features
################################################################################################

#experiments[0] = baseArgs.vary("category", "gender")
# following have not been run because it takes too long, about 3 hours for experiment 0, 1.5 hours predicted for experiment 1
#experiments[2] = baseArgs.vary("category", "long_sleeves")
#experiments[4] = baseArgs.vary("category", "jacket")
#experiments[5] = baseArgs.vary("category", "glasses")

# 100 weak: Takes about 32-50 min per run (50 if fully balanced, like gender)
baseArgs100 = baseArgs.vary("weak_count", 100)
# experiments[7]  = baseArgs100.vary("category", "gender")
# experiments[9]  = baseArgs100.vary("category", "long_sleeves")
# experiments[11] = baseArgs100.vary("category", "jacket")
# experiments[12] = baseArgs100.vary("category", "glasses")
#experiments[940]  = baseArgs100.vary("category", "random_label_per_person")


################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_YCBCR_FEATURES
################################################################################################

scaled100 = baseArgs100.vary("scale_z_to", 1.8)
# experiments[15] = scaled100.vary("category", "gender")
# experiments[17] = scaled100.vary("category", "long_sleeves")
# experiments[19] = scaled100.vary("category", "jacket")
# experiments[20] = scaled100.vary("category", "glasses")
# experiments[23] = scaled100.vary("category", "random_label_per_person")

################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_HSV_FEATURES
################################################################################################

#experiments[24] = scaled100.vary("category", "gender")
#experiments[26] = scaled100.vary("category", "long_sleeves")
#experiments[28] = scaled100.vary("category", "jacket")
#experiments[29] = scaled100.vary("category", "glasses")
#experiments[32] = scaled100.vary("category", "random_label_per_person")

################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_RGB_FEATURES
################################################################################################

#experiments[33] = scaled100.vary("category", "gender")
#experiments[35] = scaled100.vary("category", "long_sleeves")
#experiments[37] = scaled100.vary("category", "jacket")
#experiments[38] = scaled100.vary("category", "glasses")
#experiments[41] = scaled100.vary("category", "random_label_per_person")

################################################################################################
# Compile flags: With USE_EXTENT_FEATURES, no color features
################################################################################################

#experiments[42] = scaled100.vary("category", "gender")
#experiments[44] = scaled100.vary("category", "long_sleeves")
#experiments[46] = scaled100.vary("category", "jacket")
#experiments[47] = scaled100.vary("category", "glasses")
#experiments[50] = scaled100.vary("category", "random_label_per_person")


################################################################################################
# Compile flags: With USE_EXTENT_FEATURES, no color features
################################################################################################

# no point cloud scaling here
#experiments[51] = baseArgs100.vary("category", "gender")
#experiments[53] = baseArgs100.vary("category", "long_sleeves")
#experiments[55] = baseArgs100.vary("category", "jacket")
#experiments[56] = baseArgs100.vary("category", "glasses")
#experiments[59] = baseArgs100.vary("category", "random_label_per_person")


################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_YCBCR_FEATURES
################################################################################################

scaled100Seq1To3 = scaled100.vary("fold", 3000).vary("max_nan_ratio_per_volume", 0.3)
#experiments[60] = scaled100Seq1To3.vary("category", "gender")
# experiments[62] = scaled100Seq1To3.vary("category", "long_sleeves")
# experiments[64] = scaled100Seq1To3.vary("category", "jacket")
# experiments[65] = scaled100Seq1To3.vary("category", "glasses")
# experiments[68] = scaled100Seq1To3.vary("category", "random_label_per_person")


################################################################################################
# Compile flags: no USE_EXTENT_FEATURES, no color features
################################################################################################

#experiments[69] = baseArgs100.vary("category", "gender")

#experiments[70] = scaled100Seq1To3.vary("category", "gender")
# experiments[72] = scaled100Seq1To3.vary("category", "long_sleeves")
# experiments[74] = scaled100Seq1To3.vary("category", "jacket")
# experiments[75] = scaled100Seq1To3.vary("category", "glasses")
# experiments[78] = scaled100Seq1To3.vary("category", "random_label_per_person")


################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_YCBCR_FEATURES
################################################################################################

scaled100Seq1To4 = scaled100Seq1To3.vary("fold", "4000")

#experiments[79] = scaled100Seq1To4.vary("category", "gender")
#experiments[81] = scaled100Seq1To4.vary("category", "long_sleeves")
#experiments[83] = scaled100Seq1To4.vary("category", "jacket")
#experiments[84] = scaled100Seq1To4.vary("category", "glasses")
#experiments[87] = scaled100Seq1To4.vary("category", "random_label_per_person")


################################################################################################
# Compile flags: No USE_EXTENT_FEATURES, no USE_YCBCR_FEATURES
################################################################################################

base100Seq1To4 = scaled100Seq1To4.vary("scale_z_to", 0.0)

#experiments[88] = base100Seq1To4.vary("category", "gender")
#experiments[90] = base100Seq1To4.vary("category", "long_sleeves")
#experiments[92] = base100Seq1To4.vary("category", "jacket")
#experiments[93] = base100Seq1To4.vary("category", "glasses")
#experiments[96] = base100Seq1To4.vary("category", "random_label_per_person")


################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_YCBCR_FEATURES
################################################################################################

#experiments[97] = scaled100Seq1To4.vary("category", "long_trousers")
#experiments[98] = scaled100Seq1To4.vary("category", "long_hair")


################################################################################################
# Compile flags: No USE_EXTENT_FEATURES, no USE_YCBCR_FEATURES
################################################################################################

#experiments[99]  = base100Seq1To4.vary("category", "long_trousers")
#experiments[100] = base100Seq1To4.vary("category", "long_hair")

################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_YCBCR_FEATURES
################################################################################################

#experiments[101] = scaled100Seq1To4.vary("category", "long_hair").vary("crop_z_min", "1.1")



####### !!!
### NEW SEQ. 1 EXPERIMENTS FOR LONG_HAIR AND LONG_TROUSERS
####### !!!

################################################################################################
# Compile flags: no USE_EXTENT_FEATURES, no color features
################################################################################################

# 100 weak: Takes about 32-50 min per run (50 if fully balanced, like gender)
#experiments[102]  = baseArgs100.vary("category", "long_hair")
#experiments[103]  = baseArgs100.vary("category", "long_trousers")

################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_YCBCR_FEATURES
################################################################################################

#experiments[104] = scaled100.vary("category", "long_hair")
#experiments[105] = scaled100.vary("category", "long_trousers")


################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_HSV_FEATURES
################################################################################################

#experiments[106] = scaled100.vary("category", "long_hair")
#experiments[107] = scaled100.vary("category", "long_trousers")

################################################################################################
# Compile flags: With USE_EXTENT_FEATURES and USE_RGB_FEATURES
################################################################################################

#experiments[108] = scaled100.vary("category", "long_hair")
#experiments[109] = scaled100.vary("category", "long_trousers")

################################################################################################
# Compile flags: With USE_EXTENT_FEATURES, no color features
################################################################################################

#experiments[110] = scaled100.vary("category", "long_hair")
#experiments[111] = scaled100.vary("category", "long_trousers")

################################################################################################
# Compile flags: With USE_EXTENT_FEATURES, no color features
################################################################################################

# no point cloud scaling here
#experiments[112] = baseArgs100.vary("category", "long_hair")
#experiments[113] = baseArgs100.vary("category", "long_trousers")



################################################################################################
# Compile flags: With USE_EXTENT_FEATURES, no color features
################################################################################################

# Cut everything below head region (i.e. below 35 cm from top of cloud)
# No scaling since head size does not vary much!
#experiments[114] = baseArgs100.vary("category", "long_hair").vary("crop_z_min", "-0.35").vary("crop_z_max", "0")  # NEED TO RERUN DUE TO OVERWRITTEN RESULTS!

# Cut everything below waist region
#experiments[115] = scaled100.vary("category", "long_hair").vary("crop_z_min", "1.1")


################################################################################################
# Compile flags: With USE_YCBCR_FEATURES, no STANDARD_FEATURES
################################################################################################

experiments[116] = scaled100Seq1To4.vary("category", "gender")



##################################################################################################################

CSI="\x1B["
RESET_COLOR = CSI + "m"
INFO_COLOR = CSI + "7;34;47m"
SUCCESS_COLOR = CSI + "7;32;47m"
WARNING_COLOR = CSI + "7;33;47m"
ERROR_COLOR = CSI + "7;31;47m"

def printInfo(msg):
    print INFO_COLOR + msg.ljust(100) + RESET_COLOR

def printSuccess(msg):
    print SUCCESS_COLOR + msg.ljust(100) + RESET_COLOR

def printWarning(msg):
    print WARNING_COLOR + msg.ljust(100) + RESET_COLOR

def printError(msg):
    print ERROR_COLOR + msg.ljust(100) + RESET_COLOR

rospack = rospkg.RosPack()
PACKAGE_NAME = "top_down_classifier"
PACKAGE_PATH = rospack.get_path(PACKAGE_NAME)

printInfo("Preparing to run %d experiments..." % len(experiments))

experimentCounter = 0
for experimentNumber, args in experiments.iteritems():
    experimentCounter += 1

    argsStr = ""
    for argName, argValue in args.iteritems():
        argsStr += "_%s:=%s " % (argName, str(argValue))

    cmdLine = "rosrun %s train %s" % (PACKAGE_NAME, argsStr)
    runDir = "%s/results/experiment-%03d" % (PACKAGE_PATH, experimentNumber)

    logfileName = "%s/logfile.txt" % runDir

    print
    printInfo("Starting experiment %d of %d with ID %d:" % (experimentCounter, len(experiments), experimentNumber))
    printInfo("  " + cmdLine)
    printInfo("in folder: " + runDir)
    print

    if not os.path.isdir(runDir):
        os.mkdir(runDir)

    with open(logfileName, 'w') as logfile:
        logfile.write('Command-line for this experiment: \n%s\n\n' % cmdLine)
        startTime = time.time()

        try:
            # stdbuf utility prevents process from hanging. See:
            # http://stackoverflow.com/questions/20503671/python-c-program-subprocess-hangs-at-for-line-in-iter
            proc = subprocess.Popen("stdbuf -oL " + cmdLine, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=runDir, bufsize=1)

            while proc.poll() is None:
                line = proc.stdout.readline()
                if line:
                    line = line.replace(CSI+'0m', '')
                    sys.stdout.write(line)
                    logfile.write(line)
                    logfile.flush()

            duration = time.time() - startTime
            minutes = math.floor(duration / 60.0)
            seconds = duration - minutes * 60
            msg = 'Experiment %d terminated after %.0f:%02.0f min with return code %d.' %  (experimentNumber, minutes, seconds,  proc.returncode)
            logfile.write('\n' + msg)

            if proc.returncode != 0:
                printError(msg)
            else:
                printSuccess(msg)

        except OSError as e:
            msg = "Failed to run experiment %d. Reason: %s" % (experimentNumber, str(e))
            logfile.write(msg)
            printError(msg)

        except KeyboardInterrupt as e:
            print
            msg = "Interrupted by CTRL+C, terminating."
            logfile.write(msg)
            printWarning(msg)
            sys.exit(1)
