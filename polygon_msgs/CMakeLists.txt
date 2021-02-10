cmake_minimum_required(VERSION 2.8.3)
project(polygon_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  visualization_msgs
  message_generation
)


# Generating new messages
add_message_files(
  FILES 
  polygon.msg
  polygonArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  visualization_msgs
)

# catkin shit
catkin_package( CATKIN_DEPENDS message_runtime )

# Include directories
include_directories( ${catkin_INCLUDE_DIRS} )
