from utils import *
import rospy
import rospkg
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from robot_digital_twin.srv import TriggerImageSave, TriggerImageSaveRequest
from std_msgs.msg import Float64
import rosnode
import argparse

class Brick():
    def __init__(self, graph_node, lego_lib, brick_cnt):
        self.brick_id = self.read_brick_id(graph_node["brick_id"])
        self.x = graph_node["x"]
        self.y = graph_node["y"]
        self.z = graph_node["z"] + 1
        self.ori = graph_node["ori"]
        self.seq = brick_cnt[self.brick_id]
        self.height = lego_lib[str(self.brick_id)]["height"]
        self.width = lego_lib[str(self.brick_id)]["width"]
        self.name = "b" + str(self.brick_id) + "_" + str(self.seq)
    
    def read_brick_id(self, in_id):
        if(in_id in [2, 3, 4, 5, 6, 9, 10, 12]):
            return in_id
        elif(in_id == 7 or in_id == 8):
            return 5
        elif(in_id == 11):
            return 9
        return 0

class Lego():
    def __init__(self, task_fname, lego_lib, plate_x=0, plate_y=0, plate_z=0, plate_height=48, plate_width=48):
        self.plate_x = plate_x
        self.plate_y = plate_y
        self.plate_z = plate_z + 0.0016
        self.plate_height = plate_height
        self.plate_width = plate_width
        self.plate_pose = np.matrix([[1, 0, 0, self.plate_x],
                                     [0, 1, 0, self.plate_y],
                                     [0, 0, 1, self.plate_z],
                                     [0, 0, 0, 1]])
        self.brick_height_m = 0.0096
        self.P_len = 0.008
        self.brick_len_offset = 0.0002
        self.task_graph = load_json(task_fname)
        self.lego_lib = load_json(lego_lib)
        self.brick_cnt = dict()

        r1_j1_topic = "/r1/joint1_position_controller/command"
        r1_j2_topic = "/r1/joint2_position_controller/command"
        r1_j3_topic = "/r1/joint3_position_controller/command"
        r1_j4_topic = "/r1/joint4_position_controller/command"
        r1_j5_topic = "/r1/joint5_position_controller/command"
        r1_j6_topic = "/r1/joint6_position_controller/command"
        self.r1_j1_pub = rospy.Publisher(r1_j1_topic, Float64, queue_size=1)
        self.r1_j2_pub = rospy.Publisher(r1_j2_topic, Float64, queue_size=1)
        self.r1_j3_pub = rospy.Publisher(r1_j3_topic, Float64, queue_size=1)
        self.r1_j4_pub = rospy.Publisher(r1_j4_topic, Float64, queue_size=1)
        self.r1_j5_pub = rospy.Publisher(r1_j5_topic, Float64, queue_size=1)
        self.r1_j6_pub = rospy.Publisher(r1_j6_topic, Float64, queue_size=1)
        self.r1_j1_msg = Float64()
        self.r1_j2_msg = Float64()
        self.r1_j3_msg = Float64()
        self.r1_j4_msg = Float64() 
        self.r1_j5_msg = Float64()
        self.r1_j6_msg = Float64()

        r2_j1_topic = "/r2/joint1_position_controller/command"
        r2_j2_topic = "/r2/joint2_position_controller/command"
        r2_j3_topic = "/r2/joint3_position_controller/command"
        r2_j4_topic = "/r2/joint4_position_controller/command"
        r2_j5_topic = "/r2/joint5_position_controller/command"
        r2_j6_topic = "/r2/joint6_position_controller/command"
        self.r2_j1_pub = rospy.Publisher(r2_j1_topic, Float64, queue_size=1)
        self.r2_j2_pub = rospy.Publisher(r2_j2_topic, Float64, queue_size=1)
        self.r2_j3_pub = rospy.Publisher(r2_j3_topic, Float64, queue_size=1)
        self.r2_j4_pub = rospy.Publisher(r2_j4_topic, Float64, queue_size=1)
        self.r2_j5_pub = rospy.Publisher(r2_j5_topic, Float64, queue_size=1)
        self.r2_j6_pub = rospy.Publisher(r2_j6_topic, Float64, queue_size=1)
        self.r2_j1_msg = Float64()
        self.r2_j2_msg = Float64()
        self.r2_j3_msg = Float64()
        self.r2_j4_msg = Float64()
        self.r2_j5_msg = Float64()
        self.r2_j6_msg = Float64()
        
    def parse_brick(self, graph_node):
        brick_id = graph_node["brick_id"]
        if(brick_id not in self.brick_cnt.keys()):
            self.brick_cnt[brick_id] = 1
        else:
            self.brick_cnt[brick_id] += 1
       
        return Brick(graph_node, self.lego_lib, self.brick_cnt)
    
    def calc_brick_loc(self, graph_node):
        brick = self.parse_brick(graph_node)
        topleft_offset = np.identity(4)
        brick_offset = np.identity(4)
        brick_center_offset = np.identity(4)
        z_90 = np.matrix([[0, -1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        brick_offset[0, 3] = brick.x * self.P_len - self.brick_len_offset
        brick_offset[1, 3] = brick.y * self.P_len - self.brick_len_offset
        brick_offset[2, 3] = (brick.z-1) * self.brick_height_m

        brick_center_offset[0, 3] = (brick.height * self.P_len - self.brick_len_offset) / 2.0
        brick_center_offset[1, 3] = (brick.width * self.P_len - self.brick_len_offset) / 2.0
        brick_center_offset[2, 3] = 0

        topleft_offset[0, 3] = -(self.plate_height * self.P_len - self.brick_len_offset) / 2.0
        topleft_offset[1, 3] = -(self.plate_width * self.P_len - self.brick_len_offset) / 2.0
        topleft_offset[2, 3] = 0

        out_pose = self.plate_pose @ topleft_offset @ brick_offset @ brick_center_offset
        if(brick.ori == 1):
             brick_center_offset[1, 3] = -brick_center_offset[1, 3]
             out_pose = self.plate_pose @ topleft_offset @ brick_offset @ z_90 @ brick_center_offset
        return brick.name, out_pose
    
    def set_pose(self, T, name):
        x = T[0, 3]
        y = T[1, 3]
        z = T[2, 3]
        r = R.from_matrix(T[:3, :3])
        quat = r.as_quat()
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = quat[0]
        state_msg.pose.orientation.y = quat[1]
        state_msg.pose.orientation.z = quat[2]
        state_msg.pose.orientation.w = quat[3]

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)
        return resp.success
    
    def visualize(self):
        self.reset()
        self.set_pose(self.plate_pose, "assemble_plate")
        for key in self.task_graph.keys():
            node = self.task_graph[key]
            if "attached_robot_id" in node.keys():
                continue
            bname, T = self.calc_brick_loc(node)
            ret = self.set_pose(T, bname)
            if(not ret):
                print(bname, "failed!")
            time.sleep(0.1)
    
    def reset(self):
        for bid in range(13):
            for cnt in range(1, 1000):
                name = "b" + str(bid) + "_" + str(cnt)
                ret = self.set_pose(np.identity(4), name)
                if(not ret):
                    self.brick_cnt[bid] = 0
                    break
    
    def update_robot(self, pose1, pose2):
        i = 0
        while i < 20:
            i += 1
            self.r1_j1_msg.data = pose1[0]
            self.r1_j2_msg.data = pose1[1]
            self.r1_j3_msg.data = pose1[2]
            self.r1_j4_msg.data = pose1[3]
            self.r1_j5_msg.data = pose1[4]
            self.r1_j6_msg.data = pose1[5]
            self.r1_j1_pub.publish(self.r1_j1_msg)
            self.r1_j2_pub.publish(self.r1_j2_msg)
            self.r1_j3_pub.publish(self.r1_j3_msg)
            self.r1_j4_pub.publish(self.r1_j4_msg)
            self.r1_j5_pub.publish(self.r1_j5_msg)
            self.r1_j6_pub.publish(self.r1_j6_msg)

            self.r2_j1_msg.data = pose2[0]
            self.r2_j2_msg.data = pose2[1]
            self.r2_j3_msg.data = pose2[2]
            self.r2_j4_msg.data = pose2[3]
            self.r2_j5_msg.data = pose2[4]
            self.r2_j6_msg.data = pose2[5]
            self.r2_j1_pub.publish(self.r2_j1_msg)
            self.r2_j2_pub.publish(self.r2_j2_msg)
            self.r2_j3_pub.publish(self.r2_j3_msg)
            self.r2_j4_pub.publish(self.r2_j4_msg)
            self.r2_j5_pub.publish(self.r2_j5_msg)
            self.r2_j6_pub.publish(self.r2_j6_msg)
            time.sleep(0.1)
    
    def save_imgs(self, prefix):
        rospy.wait_for_service('/save_gazebo_images')
        save_service = rospy.ServiceProxy('/save_gazebo_images', TriggerImageSave)
        req = TriggerImageSaveRequest()
        req.base_save_path = "/home/philip/gazebo_images/{}".format(prefix)
        print("Saving images to: ", req.base_save_path)

        resp = save_service(req)
        if resp.success:
            rospy.loginfo("Images saved successfully.")
            print(resp.image_path_cam1, resp.image_path_cam2)
        else:
            rospy.logerr("Failed to save images.")
            print(req.message)

if __name__ == '__main__':
    rospy.init_node('vis_lego')
    rospy.sleep(1)
    
    # the task is xxxx.json, read xxxx from the command line
    parser = argparse.ArgumentParser(description='Visualize Lego task graph')
    parser.add_argument('task', type=str, help='Task name')
    args = parser.parse_args()

    task_fname = "./scripts/tasks/" + args.task + ".json"
    lego = Lego(task_fname, "./scripts/lego_library.json",
     plate_x=0.40842053781513565, 
     plate_y=0.04519264491562785,
     plate_z=0.1899+0.926)
    
    home_pose = np.array([0.0,-15.456,-40.357,0,-65.099,0]) / 180.0 * np.pi
    lego.update_robot(home_pose, home_pose)
    
    lego.visualize()
    rospy.sleep(4)

    lego.save_imgs(args.task)