cmake_minimum_required(VERSION 3.1)
project(active_grasp)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  std_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  AABBox.msg
)

add_service_files(
  FILES
  Reset.srv
  Seed.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)
