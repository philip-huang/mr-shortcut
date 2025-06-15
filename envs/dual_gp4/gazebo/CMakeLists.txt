cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
project(robot_digital_twin VERSION 1.0.0 LANGUAGES CXX)

# /* -------------------------------------------------------------------------- */
# /*                              Build Properties                              */
# /* -------------------------------------------------------------------------- */

add_compile_options(-std=c++11)
add_definitions(-D_OS_UNIX)

# /* -------------------------------------------------------------------------- */
# /*                                Find Package                                */
# /* -------------------------------------------------------------------------- */

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

add_service_files(
  FILES
  TriggerImageSave.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime # Add message_runtime\
)
include_directories(
    ${catkin_INCLUDE_DIRS})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})