cmake_minimum_required(VERSION 2.8.3)
project(au_path)


find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  cv_bridge
  image_transport
  dynamic_reconfigure
  actionlib
  #au_core
  #au_pid
)

## System dependencies are found with CMake's conventions
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(catkin REQUIRED genmsg actionlib_msgs actionlib)

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  cfg/path_para.cfg
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
   INCLUDE_DIRS
   LIBRARIES au_path
   CATKIN_DEPENDS roscpp std_msgs rospy #actionlib
   DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include/au_path
)

## Declare a C++ library
# add_library(au_path
#   src/${PROJECT_NAME}/au_path.cpp
# )
## Declare a C++ executable
add_executable(au_path_node src/connections.cpp src/imageprocessing.h)
add_executable(testing_node src/testing.cpp src/imageprocessing.h)
target_link_libraries(au_path_node
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
)
target_link_libraries(testing_node
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
)
add_dependencies(au_path_node ${PROJECT_NAME}_gencfg)
add_dependencies(testing_node ${PROJECT_NAME}_gencfg)


