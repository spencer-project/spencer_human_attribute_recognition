### Tessellation-based RGB-D point cloud classifier for real-time, full-body human attribute recognition ###

![Teaser Image 1](/../screenshots/screenshots/iros15-teaser-1.png?raw=true "Teaser Image 1") &nbsp;&nbsp;&nbsp;&nbsp; ![Teaser Image 2](/../screenshots/screenshots/teaser-2.png?raw=true "Teaser Image 2")


#### Introduction #####

This repository contains a ROS-based C++ / Python implementation of the classification approach described in the paper

> *Real-Time Full-Body Human Attribute Classification in RGB-D Using a Tessellation Boosting Approach*  
> by Timm Linder and Kai O. Arras   
> IEEE/RSJ Int. Conference on Intelligent Robots and Systems (IROS), Hamburg, Germany, 2015.

Please cite our paper if you use this code in your research.

The depth-based tessellation learning method described in this paper is an extension of a top-down classifier for people detection, originally described in

> *Tracking People in 3D Using a Bottom-Up Top-Down Detector*  
> by Luciano Spinello, Matthias Luber, Kai O. Arras    
> IEEE International Conference on Robotics and Automation (ICRA'11), Shanghai, China, 2011.    

This complete re-implementation of the original code, among other optimizations, adds further tessellation scales and aspect ratios, new geometric extent and color features in different color spaces, filtering of low-quality training samples, and supports use of ROS/Rviz for visualization. It uses the Adaboost implementation from the OpenCV machine learning module.
Point cloud files are loaded and pre-processed using the Point Cloud Library (PCL).


#### Installation and setup ####

##### System requirements #####

* Quad-core CPU recommended for efficient training/testing
* 16 GB of RAM (for training, on the full dataset)
* Around 300 GB of hard disk space for storing the entire dataset (>100 persons, 500-800 point cloud frames per person). SSD highly recommended.
* Point Cloud Library (PCL) 1.7
* OpenCV 2.3.1
* Eigen3

This package has only been tested on Ubuntu Trusty 14.04 LTS (64-Bit) with ROS Indigo.

Most of the required dependencies are installed automatically with ROS (`sudo apt-get install ros-hydro-desktop-full`).
To install Eigen3, use `sudo apt-get install libeigen3-dev`.


##### How to build #####

Once all of the above dependencies are installed, follow these steps:

1. Create a new ROS workspace as described in [this tutorial](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)
2. Clone the content of this repository into the `src` folder of your workspace.
3. Build the workspace by running `catkin_make -DCMAKE_BUILD_TYPE=RelWithDebInfo`


#### Dataset ####

##### Dataset format #####

The experiments in our paper were conducted using the point cloud version of the SRL Human Attribute Dataset. To request access to this dataset for research purposes, [visit this page](http://srl.informatik.uni-freiburg.de/human_attributes_dataset).


For our experiments, we only used the higher-resolution Kinect v2 data. The code in this ROS package assumes that the data is stored on hard disk in the following layout:

    srl_human_attribute_dataset/
        kinect2/
            clouds/
                person_XXX/
                    person_XXX_S_F_cloud.pcd
                    person_XXX_S_F_pose.txt

Here, `XXX` is the person number, `S` the sequence number (between 1 and 4; 1: standing persons in 8 different orientations, 2: complex walking pattern, 3: circular walk, 4: close-up interaction),
and `F` the frame number. For sequence 1 in our dataset, frames 1-8 correspond to 8 discrete orientations in 45 degree angles, but this is not mandatory for custom datasets.

For each person instance, there must be a separate directory containing all the point cloud frames (in PCL's binary PCD format) and associated groundtruth poses (only used for analysis,
and to filter out low-quality point clouds during training). The point cloud must be pre-segmented and only contain the person itself, but no background or ground plane (as far as possible,
within given noise margins). The person shall be centered on the origin in `X` and `Y`, and `Z = 0` shall be the location of the person's feet. For an example of a correct positioning and orientation,
see the `test_cloud.pcd` file in the `top_down_classifier/data` folder.

The associated `_pose.txt` files shall contain the following lines (example values):

    position_x  2.4183
    position_y  -0.107292
    velocity_x  0
    velocity_y  0
    theta_deg   180
    sensor_distance 2.42068

Position and sensor distance are used during training to filter out point clouds which are too close to the sensor's near clipping plane.


##### Train/test split format #####

Since recording RGB-D point clouds of a large number of real persons under different orientations, distances and walking patterns is a very time-consuming task,
it is not easily possible to cover thousands of persons in a single dataset. As the current dataset used in our research, the SRL Human Attribute Dataset, contains
'only' slightly over 100 person (making it the largest such RGB-D person dataset that we are aware of), using k-fold cross-validation for training is not a good idea
since the folds quickly become too small, which leads to overfitting.

Instead, for our experiments, we use repeated random subsampling validation: We randomly divide the dataset, on a per-person basis, into two (usually equally sized) parts,
a training and a test set. The same person instance never occurs in both the training and test set simultaneously (i.e. we ensure that all point cloud frames of the same instance
end up in the same set). We repeat this process k times, leading to k folds each with different randomized train/test splits. We then train on each fold individually,
by using the k-th training set, evaluate the classifier performance on the corresponding k-th test set, and in the end average the accuracy over all k test sets. We
save the classifier model that had the highest overall accuracy on its corresponding test set.

Train/test splits (folds) are generated by iterating over all the point cloud files in the dataset, and a groundtruth file containing person labels (gender, has-long-trousers, etc.) using the `generate_pose_labels_clouds.py` script inside the `top_down_classifier/scripts` folder. It needs to be run once for each fold (e.g. fold 100, 101, 102, ...) of each human attribute;
in our experiments, the fold number also corresponds to the random seed, which can be provided to the script as the first command-line argument. Other parameters, such as which
dataset sequences to use (1 to 4) and which human attribute to consider, must be changed inside the Python script.

The resulting train and test splits for the current fold are called `train.txt` and `val.txt`. These are simple text files, each line containing a file path to a PCD file of the dataset, and a
label (0 for negative class, 1 for positive class). Each file starts with two or three lines of comments, indicated by `#`. Train/test splits should be stored in the following directory structure:

    top_down_classifier/
       data/
            attribute_name/
                fold100/
                    train.txt
                    val.txt
                fold101/
                    train.txt
                    val.txt

Please note that the paths to the point clouds in the provided train/test splits inside the `data` folder are currently hard-coded. If you are storing the
dataset in a different location, we recommend using a tool such as regexxer or sed to replace all the paths.


#### Usage ####

##### Visualization of tessellations and person clouds #####

To visualize the person point clouds processed during training, and the resulting best tessellation, run
    roslaunch top_down_classifier visualization.launch

This automatically runs an instance of RViz, with the correct displays, topics and transforms set up to show the visualization results published by the `train` ROS node.

To instead view a single PCD (point cloud) file, run

    roslaunch pcd_viewer pcd_viewer.launch filename:=~/srl_human_attribute_dataset/kinect2/clouds/person_XXX/person_XXX_S_F_cloud.pcd

where `XXX` is the person number, `S` the sequence number and `F` the frame number. See the dataset documentation for details. This launches a separate
instance of Rviz, which just displays the point cloud in depth+color and depth-only.


##### Training a classifier #####

Most code inside the `top_down_classifier` ROS package deals with learning a classifier, by selecting the best features, thresholds and tessellation volumes in which to compute these features.
All the training (and validation testing) happens inside a ROS node called `train`.

Example command line:

    rosrun top_down_classifier train _category:=long_trousers _fold:=100 _num_folds:=10 _weak_count:=100

Overview of the most important command-line parameters with some sensible default values:

* `_category:=long_trousers`: The name of the human attribute to consider (gender, long_hair, etc). Must correspond to the name of one of the subfolders in the `data` directory.

* `_fold:=100`: The train/test split to start training with. There must be a corresponding foldXXX folder in the `data` directory containing a `train.txt` and `val.txt` file, as described above.

* `_num_folds:=10`: The number of train/test splits to process. If e.g. you specify `_fold:=100 _num_folds:=10`, there must be folders `fold100` to `fold109` inside the `data` directory.

* `_weak_count:=100`: The number of weak classifiers (decision stumps) to learn using Adaboost. Recommended minimum value is 100, at 500 the results are in average 1.5% better, even larger values do not bring much further change. The more weak classifiers, the longer the training takes and the slower the final classifier during testing.

* `_show_tessellations:=false`: If set to true, all generated tessellations (according to the hard-coded scales and aspect ratios) are published as visualization markers (which can be displayed in Rviz) before starting the training process.

* `_interactive:=false`: If true, and `_show_tessellations` is also true, the user needs to hit a key after a generated tessellation has been published for visualization, in order to proceed.

* `_regular_tessellation_only:=false`: If true, only a single regular tessellation (with cubic voxels of size `_regular_tessellation_size` in meters) will be generated and used for training. Only useful to compare against learned tessellations, which are used by default.

* `_scale_z_to:=0.0`: Height (in meters) to which the point cloud should be scaled. 0.0 means no scaling in `z` direction. See the paper to understand the effect of this parameter.

* `_min_voxel_size:=0.1`: Minimum size of the cubic voxels (in meters) used to generate tessellations of the bounding volume. Smaller voxels means longer feature descriptors and higher RAM usage, but could help to capture finer details (if there is still a sufficient number of points per voxel).

At the end of the training process, the best learned classifier (across all folds) is stored in YAML format as `top_down_classifier.yaml` inside the current working directory. The classifier is saved using
the OpenCV machine learning APIs, along with some meta-data.

For each train/test split, the `train` ROS node also saves a CSV file including prediction result and person pose/orientation (extracted from the dataset's `_pose.txt` files) inside the current working directory, which are afterwards useful for more detailed analysis.


##### Optimizing a learned classifier #####

The `top_down_classifier.yaml` file saved at the end of the training process by the `train` node still contains information about all the originally generated tessellation volumes and features. This usually leads to an over 120,000-dimensional feature vector for each sample. However, since Adaboost automatically learns the n best weak classifiers, most tessellation volumes and features calculated inside these volumes are actually never used by the resulting strong classifier. For this purpose, we can get rid of any unnecessary tessellation volumes and features by analyzing the weak classifiers in the `top_down_classifier.yaml` file. This is done by the script `optimize_classifier_model.py`:

    rosrun top_down_classifier optimize_classifier_model.py top_down_classifier.yaml

The output is a significantly smaller `top_down_classifier_optimized.yaml` model, which when used for classification (see the following step) usually leads to a feature vector of less than 1% of its original size. This significantly reduces the memory consumption and processing time of the classifier without any negative impact on classification performance.


##### Testing a learned classifier #####

To test an optimized or unoptimized classifier model (loaded from a YAML file) on a single train or test split, run:

    rosrun top_down_classifier test_classifier _model:=top_down_classifier_optimized.yaml _list_file:=`rospack find top_down_classifier`/data/attribute_name/foldXXX/val.txt

This ROS node will output the resulting accuracy to the console, and also publish the learned best tessellation contained in this YAML file as a `visualization_msgs/MarkerArray` that can be displayed in Rviz.


##### Applying a learned classifier to a single point cloud #####

To classify a single point cloud (PCD file) using an existing classifier model, run e.g.:

    rosrun top_down_classifier classify_single_cloud _filename:=~/srl_human_attribute_dataset/kinect2/clouds/person_XXX/person_XXX_S_F_cloud.pcd rviz:=false _model:=`rospack find top_down_classifier`/learned_models/top_down_classifier_fold207_optimized.yaml

The predicted class label (for gender: 0=female, 1=male) is output to the console and also corresponds to the return value of the process (or -1 if an error occurred).


#### Credits ####

Code written by Timm Linder, Social Robotics Laboratory, University of Freiburg.

We would like to thank Dr. Luciano Spinello for kindly providing us parts of the original detector code from the paper *Tracking People in 3D Using a Bottom-Up Top-Down Detector* by *L. Spinello, M. Luber and K.O. Arras*. Parts of this code were used in the implementation of `tessellation_generator.cpp`.


#### License ####

Software License Agreement (BSD License)

Copyright (c) 2014-2015, Timm Linder, Social Robotics Lab, University of Freiburg
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
