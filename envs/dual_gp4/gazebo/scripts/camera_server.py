#!/usr/bin/env python

import rospy
import cv2
import os
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from robot_digital_twin.srv import TriggerImageSave, TriggerImageSaveResponse # CHANGE my_package

class GazeboImageSaver:
    def __init__(self):
        rospy.init_node('gazebo_image_saver', anonymous=True)

        # --- Configuration ---
        # Get camera topics from parameters or use defaults
        self.camera_topic_1 = rospy.get_param('~camera_topic_1', '/camera1/fixed_camera/image_raw')
        self.camera_topic_2 = rospy.get_param('~camera_topic_2', '/camera2/fixed_camera/image_raw')
        self.save_dir = rospy.get_param('~save_dir', os.path.expanduser('~/gazebo_images'))
        self.cam1_name = rospy.get_param('~cam1_name', 'cam1')
        self.cam2_name = rospy.get_param('~cam2_name', 'cam2')
        # --- End Configuration ---

        self.bridge = CvBridge()
        self.latest_image_cam1 = None
        self.latest_image_cam2 = None
        self.image_received_cam1 = False
        self.image_received_cam2 = False
        self.save_counter = 0

        # Ensure save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            rospy.loginfo("Created save directory: {}".format(self.save_dir))

        # Subscribers
        self.image_sub_cam1 = rospy.Subscriber(self.camera_topic_1, Image, self.image_callback_cam1)
        self.image_sub_cam2 = rospy.Subscriber(self.camera_topic_2, Image, self.image_callback_cam2)
        rospy.loginfo("Subscribed to {} for {}".format(self.camera_topic_1, self.cam1_name))
        rospy.loginfo("Subscribed to {} for {}".format(self.camera_topic_2, self.cam2_name))


        # Service Server
        self.save_service = rospy.Service('save_gazebo_images', TriggerImageSave, self.handle_save_request)
        rospy.loginfo("Image saving service '/save_gazebo_images' ready.")

    def image_callback_cam1(self, data):
        self.latest_image_cam1 = data
        if not self.image_received_cam1:
            self.image_received_cam1 = True
            rospy.loginfo("First image received from {}".format(self.cam1_name))

    def image_callback_cam2(self, data):
        self.latest_image_cam2 = data
        if not self.image_received_cam2:
            self.image_received_cam2 = True
            rospy.loginfo("First image received from {}".format(self.cam2_name))

    def handle_save_request(self, req):
        rospy.loginfo("Save request received.")
        response = TriggerImageSaveResponse()
        response.success = False # Default to failure

        if not self.image_received_cam1 or not self.image_received_cam2:
            response.message = "Images not yet received from both cameras."
            rospy.logwarn(response.message)
            return response

        # Use copies to avoid issues if callbacks update during saving
        img_msg_cam1 = self.latest_image_cam1
        img_msg_cam2 = self.latest_image_cam2

        try:
            # Convert ROS Image messages to OpenCV images
            cv_image_cam1 = self.bridge.imgmsg_to_cv2(img_msg_cam1, "bgr8")
            cv_image_cam2 = self.bridge.imgmsg_to_cv2(img_msg_cam2, "bgr8")
        except CvBridgeError as e:
            response.message = "CvBridge Error: {}".format(e)
            rospy.logerr(response.message)
            return response

        # --- Filename Generation ---
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        seq_str = "{:04d}".format(self.save_counter)

        # Determine save directory from request or default
        current_save_dir = self.save_dir
        if req.base_save_path:
             # Treat request path as directory if it ends with / or is an existing dir
             if req.base_save_path.endswith('/') or os.path.isdir(req.base_save_path):
                 current_save_dir = req.base_save_path
                 if not os.path.exists(current_save_dir):
                     try:
                         os.makedirs(current_save_dir)
                         rospy.loginfo("Created requested save directory: {}".format(current_save_dir))
                     except OSError as e:
                         response.message = "Could not create requested directory {}: {}".format(current_save_dir, e)
                         rospy.logerr(response.message)
                         return response
             else:
                # Use request as a base filename prefix
                base_name = os.path.basename(req.base_save_path)
                current_save_dir = os.path.dirname(req.base_save_path)
                if not current_save_dir: # Handle case where only filename is given
                    current_save_dir = self.save_dir
                if not os.path.exists(current_save_dir):
                     try:
                         os.makedirs(current_save_dir)
                         rospy.loginfo("Created directory for requested base name: {}".format(current_save_dir))
                     except OSError as e:
                         response.message = "Could not create directory for requested base name {}: {}".format(current_save_dir, e)
                         rospy.logerr(response.message)
                         return response
                # Override default naming pattern if base_save_path looks like a filename prefix
                filename_cam1 = os.path.join(current_save_dir, "{}_{}.png".format(base_name, self.cam1_name))
                filename_cam2 = os.path.join(current_save_dir, "{}_{}.png".format(base_name, self.cam2_name))
                # Avoid using timestamp/counter if specific base name provided
                timestamp = "" # Clear timestamp/seq if base name is used
                seq_str = "" #


        # Default filename pattern if not overridden by base_save_path prefix
        if not ('filename_cam1' in locals() and 'filename_cam2' in locals()):
            filename_cam1 = os.path.join(current_save_dir, "{}_{}_{}.png".format(self.cam1_name, timestamp, seq_str))
            filename_cam2 = os.path.join(current_save_dir, "{}_{}_{}.png".format(self.cam2_name, timestamp, seq_str))


        # --- Saving ---
        try:
            save_ok_1 = cv2.imwrite(filename_cam1, cv_image_cam1)
            save_ok_2 = cv2.imwrite(filename_cam2, cv_image_cam2)

            if save_ok_1 and save_ok_2:
                response.success = True
                response.message = "Images saved successfully."
                response.image_path_cam1 = filename_cam1
                response.image_path_cam2 = filename_cam2
                rospy.loginfo("Saved: {}".format(filename_cam1))
                rospy.loginfo("Saved: {}".format(filename_cam2))
                self.save_counter += 1
            else:
                err_msg = []
                if not save_ok_1: err_msg.append("Failed to save {}".format(self.cam1_name))
                if not save_ok_2: err_msg.append("Failed to save {}".format(self.cam2_name))
                response.message = "; ".join(err_msg)
                rospy.logerr(response.message)

        except Exception as e:
            response.message = "Error during saving: {}".format(e)
            rospy.logerr(response.message)
            response.success = False # Ensure success is false on exception

        return response

if __name__ == '__main__':
    try:
        saver = GazeboImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass