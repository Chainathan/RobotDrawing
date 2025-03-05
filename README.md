### Useful Links 

1. https://discuss.luxonis.com/d/5129-transforming-oak-d-lite-frame-to-panda-arm-frame
2. https://docs.luxonis.com/software/ros/depthai-ros/driver/
3. 


### To Setup Kinect v1 Camera : 

Make sure to connect camera to USB 3.0 Port.

cd kinect_ros2

source install/setup.bash

To view RGB Feed : 

ros2 launch kinect_ros2 showimage.launch.py


### List of Topics from OAK-D Lite and WX250s

/attached_collision_object

/clicked_point

/clock

/collision_object

/diagnostics

/display_contacts

/display_planned_path

/goal_pose

/initialpose

/joint_states

/monitored_planning_scene

/motion_plan_request

/oak/imu/data

/oak/nn/spatial_detections

/oak/rgb/camera_info

/oak/rgb/image_raw

/oak/rgb/image_raw/compressed

/oak/rgb/image_raw/compressedDepth

/oak/rgb/image_raw/theora

/oak/rgb/image_rect

/oak/rgb/image_rect/compressed

/oak/rgb/image_rect/compressedDepth

/oak/rgb/image_rect/theora

/oak/stereo/camera_info

/oak/stereo/image_raw

/oak/stereo/image_raw/compressed

/oak/stereo/image_raw/compressedDepth

/oak/stereo/image_raw/theora

/parameter_events

/planning_scene

/planning_scene_world

/recognized_object_array

/robot_description

/rosout

/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback

/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update

/tf

/tf_static

/trajectory_execution_event

/wx250s/joint_states
/wx250s/robot_description
