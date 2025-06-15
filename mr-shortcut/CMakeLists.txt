cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
project(mr-shortcut VERSION 1.0.0 LANGUAGES CXX)

# /* -------------------------------------------------------------------------- */
# /*                              Build Properties                              */
# /* -------------------------------------------------------------------------- */
add_compile_options(-std=c++11)
add_definitions(-D_OS_UNIX)
# Check if profiling is needed
option(ENABLE_PROFILING "Enable profiling with gprof" OFF)

if (ENABLE_PROFILING)
    # Add the -pg flag to the compiler and linker flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()


# /* -------------------------------------------------------------------------- */
# /*                                Find Package                                */
# /* -------------------------------------------------------------------------- */
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread serialization)
find_package(ompl REQUIRED)
find_package(OpenMP REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  rviz_visual_tools
  moveit_visual_tools
  tf2_geometry_msgs
  message_generation
  actionlib
)


# /* -------------------------------------------------------------------------- */
# /*                               Catkin Package                               */
# /* -------------------------------------------------------------------------- */
catkin_package(
  LIBRARIES
  INCLUDE_DIRS
  CATKIN_DEPENDS
    moveit_core
    moveit_visual_tools
    moveit_ros_planning_interface
    tf2_geometry_msgs 
    message_runtime
  DEPENDS
    EIGEN3
)

# /* -------------------------------------------------------------------------- */
# /*                               Include Directories                          */
# /* -------------------------------------------------------------------------- */

include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})

# /* -------------------------------------------------------------------------- */
# /*                               Create Library                               */
# /* -------------------------------------------------------------------------- */
file(GLOB src_files src/instance.cpp src/planner.cpp src/SingleAgentPlanner.cpp
                    src/logger.cpp src/tpg.cpp 
                    src/shortcutter.cpp)

add_library(${PROJECT_NAME} SHARED ${src_files})

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  ${catkin_LIBRARIES}
  ${OMPL_LIBRARIES}
  ${Boost_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE src)

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1)




# /* -------------------------------------------------------------------------- */
# /*                             Testing Executables                            */
# /* -------------------------------------------------------------------------- */


add_executable(demo
  src/exe/demo_node.cpp
)


target_link_libraries(demo
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

install(TARGETS demo
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)