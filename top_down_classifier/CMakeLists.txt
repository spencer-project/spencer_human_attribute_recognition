cmake_minimum_required(VERSION 2.8.3)
project(top_down_classifier)

find_package(OpenCV REQUIRED)
message(STATUS "Using OpenCV version ${OpenCV_VERSION} installed at: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV supported compute capabilities are: ${OpenCV_COMPUTE_CAPABILITIES}")

find_package(PCL REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(Boost REQUIRED COMPONENTS )
find_package(OpenCV REQUIRED core imgproc ml highgui)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp roslib pcl_conversions pcl_ros eigen_conversions tf_conversions visualization_msgs cmake_modules
  # spencer_tracking_msgs spencer_vision_msgs

  # tf_conversions has a dependency on kdl, but doesn't pull it in correctly (might get linker error without this)
  orocos_kdl
)



catkin_package()


include_directories(include src ${catkin_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


add_library(cnpy
    src/top_down_classifier/3rd_party/cnpy/cnpy.cpp)


add_executable(train
    src/top_down_classifier/features.cpp
    src/top_down_classifier/volume.cpp
    src/top_down_classifier/volume_visualizer.cpp
    src/top_down_classifier/tessellation_generator.cpp
    src/top_down_classifier/nodes/train.cpp)

add_dependencies(train ${catkin_EXPORTED_TARGETS}) # for generated message headers etc.
target_link_libraries(train cnpy ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(test_features
    src/top_down_classifier/features.cpp
    src/top_down_classifier/volume.cpp
    src/top_down_classifier/nodes/test_features.cpp)

add_dependencies(test_features ${catkin_EXPORTED_TARGETS})
target_link_libraries(test_features ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(test_classifier
    src/top_down_classifier/features.cpp
    src/top_down_classifier/volume.cpp
    src/top_down_classifier/volume_visualizer.cpp
    src/top_down_classifier/classifier.cpp
    src/top_down_classifier/nodes/test_classifier.cpp)

add_dependencies(test_classifier ${catkin_EXPORTED_TARGETS}) # for generated message headers etc.
target_link_libraries(test_classifier ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(classify_single_cloud
    src/top_down_classifier/features.cpp
    src/top_down_classifier/volume.cpp
    src/top_down_classifier/classifier.cpp
    src/top_down_classifier/nodes/classify_single_cloud.cpp)

add_dependencies(classify_single_cloud ${catkin_EXPORTED_TARGETS}) # for generated message headers etc.
target_link_libraries(classify_single_cloud ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
