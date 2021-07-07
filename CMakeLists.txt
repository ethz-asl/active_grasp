cmake_minimum_required(VERSION 3.1)
project(active_grasp)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  AABBox.msg
)

add_service_files(
  FILES
  Reset.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)
