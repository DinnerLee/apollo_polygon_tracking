cmake_minimum_required(VERSION 2.8.3)
project(apollo_polygon_tracking)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11")
###CAFFE
set(CAFFE_PATH "$ENV{HOME}/caffe/distribute")

find_package(CUDA)

include_directories(${Driveworks_INCLUDE_DIR})
list(APPEND Driveworks_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
list(APPEND Driveworks_LIBRARIES)

find_library(DRIVEWORKS_FOUND NAMES driveworks)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(DRIVEWORKS_LINKER_FLAGS "-Wl,--unresolved-symbols=ignore-in-object-files -Wl,--allow-shlib-undefined")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(DRIVEWORKS_LINKER_FLAGS "-Wl,-undefined,dynamic_lookup")
    endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rockauto_build_flags
  cv_bridge
  roscpp
  std_msgs
  image_transport
  pcl_ros
  sensor_msgs
  geometry_msgs
  rockauto_msgs
  polygon_msgs
  #msgs_package
)
find_package(OpenCV)

catkin_package(
 INCLUDE_DIRS
 CATKIN_DEPENDS
 cv_bridge
 roscpp
 std_msgs
 pcl_ros
 sensor_msgs
)

add_definitions(-LINUX)

include_directories(
  ${catkin_INCLUDE_DIRS}
  /usr/local/driveworks/include
  seg_cluster
  Association
  ${OpenCV_INCLUDE_DIRS}
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CAFFE_PATH}/include
)

link_directories(
   ${catkin_LIBRARY_DIRS}
   /usr/lib/
   /usr/local/driveworks/lib/
)

cuda_add_executable(${PROJECT_NAME}_node
 main.cpp

 seg_cluster/include/disjoint_set.h
 seg_cluster/include/pcl_types.h
 seg_cluster/include/util.h
 seg_cluster/include/cnn_segmentation.h
 seg_cluster/include/cluster2d.h
 seg_cluster/include/feature_generator.h
 seg_cluster/src/feature_generator.cpp
 seg_cluster/src/cluster2d.cpp
 seg_cluster/src/cnn_segmentation.cpp

 Association/include/TrackAssociation.h
 Association/src/TrackAssociation.cpp
)

IF (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})

    IF ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "^arm")
        link_directories(/usr/lib/arm-linux-gnueabihf/tegra)
    endif ()
ENDIF ()

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  driveworks
  ${DRIVEWORKS_LINKER_FLAGS}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CAFFE_PATH}/lib/libcaffe.so
  glog
)

add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
