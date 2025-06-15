#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

#include "tpg.h"
#include "planner.h"
#include "logger.h"
#include "shortcutter.h"
 
class TestPPPlanning {
public:
    TestPPPlanning(ros::NodeHandle &nh,
                    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group,
                    planning_scene::PlanningScenePtr &planning_scene,
                    robot_state::RobotStatePtr  &kinematic_state,
                    const std::vector<std::string> &group_names,
                    const std::vector<std::string> &eof_groups,
                    const std::string &planner_name,
                    double planning_time_limit,
                    bool async,
                    bool mfi,
                    bool load_tpg,
                    bool benchmark,
                    bool load_cbs,
                    bool chomp,
                    bool tpg_shortcut,
                    bool build_tpg,
                    const ShortcutOptions &sc_options,
                    const std::string &benchmark_fname,
                    const std::string &tpg_savedir) : nh(nh), move_group(move_group), planning_scene(planning_scene), 
                kinematic_state(kinematic_state), group_names(group_names), eof_groups(eof_groups), async(async),
                planning_time_limit(planning_time_limit), mfi(mfi), planner_name(planner_name),
                load_tpg(load_tpg), benchmark(benchmark), benchmark_fname(benchmark_fname),
                chomp(chomp), tpg_shortcut(tpg_shortcut), sc_options(sc_options), build_tpg(build_tpg),
                tpg_savedir(tpg_savedir), load_cbs(load_cbs)
        {
            num_robots = group_names.size();

            planning_scene_diff_client = nh.serviceClient<moveit_msgs::ApplyPlanningScene>("apply_planning_scene");
            planning_scene_diff_client.waitForExistence();
        
            instance_ = std::make_shared<MoveitInstance>(kinematic_state, move_group->getName(), planning_scene);
            instance_->setNumberOfRobots(num_robots);
            instance_->setRobotNames(group_names);
            if (eof_groups.size() > 0) {
                instance_->setHandNames(eof_groups);
            }
            for (int i = 0; i < num_robots; i++) {
                instance_->setRobotDOF(i, 7);
            }
            instance_->setVmax(1.0);
            instance_->setPlanningSceneDiffClient(planning_scene_diff_client);
        }

    void setup_once() {
        /*
        Set joint name and record start locations. Necessary for execution
        */
        joint_names = move_group->getVariableNames();
        joint_names_split.clear();
        left_arm_joint_state_received = false;
        right_arm_joint_state_received = false;

        // create a directory for saving TPGs if it does not exist
        if (!boost::filesystem::exists(tpg_savedir)) {
            boost::filesystem::create_directories(tpg_savedir);
        }

        if (benchmark) {
            current_joints = move_group->getCurrentJointValues();
            // clear the benchmark file
            logProgressFileStart(benchmark_fname);
        }
        else if (mfi) {
            std::vector<std::string> names;
            names.push_back("joint_1_s");
            names.push_back("joint_2_l");
            names.push_back("joint_3_u");
            names.push_back("joint_4_r");
            names.push_back("joint_5_b");
            names.push_back("joint_6_t");
            joint_names_split.push_back(names);
            joint_names_split.push_back(names);
            current_joints.resize(14, 0.0);
           
            left_arm_sub = nh.subscribe("/yk_destroyer/joint_states", 1, &TestPPPlanning::left_arm_joint_state_cb, this);
            right_arm_sub = nh.subscribe("/yk_architect/joint_states", 1, &TestPPPlanning::right_arm_joint_state_cb, this);

            while (!left_arm_joint_state_received || !right_arm_joint_state_received) {
                ros::Duration(0.1).sleep();
            }
        }
        else {
            int num_robots = instance_->getNumberOfRobots();
            current_joints.resize(num_robots * 7, 0.0);
            for (int i = 0; i < num_robots; i++) {
                std::vector<std::string> name_i;
                for (size_t j = 0; j < 7; j++) {
                    name_i.push_back(joint_names[i*7+j]);
                }
                joint_names_split.push_back(name_i);
            }
            all_arm_sub = nh.subscribe("joint_states", 1, &TestPPPlanning::all_arm_joint_state_cb, this);
            while (!left_arm_joint_state_received || !right_arm_joint_state_received) {
                ros::Duration(0.1).sleep();
            }
        }
    }

    bool saveTPG(const std::string &filename) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            ROS_ERROR("Failed to open file when saving TPG: %s", filename.c_str());
            return false;
        }
        boost::archive::text_oarchive oa(ofs);
        oa << tpg;
        
        return true;
    }

    bool loadTPG(const std::string &filename) {
        // open the file safely
        std::ifstream ifs(filename);
        ROS_INFO("Loading TPG from file: %s", filename.c_str());
        if (!ifs.is_open()) {
            ROS_ERROR("Failed to open file when loading TPG: %s", filename.c_str());
            return false;
        }
        
        boost::archive::text_iarchive ia(ifs);
        ia >> tpg;
        return true;
    }

    bool motion_plan(const std::string &pose_name, std::shared_ptr<MoveitInstance> instance, const tpg::TPGConfig &tpg_config, int recursion = 0) {
        /*
        Call the planner
        */
        if (recursion >= 5) {
            ROS_ERROR("Failed to plan after 5 attempts");
            return false;
        }
        MRTrajectory solution;
        auto tic = std::chrono::high_resolution_clock::now();
        t_plan = 0.0;
        if (planner_name == "PrioritizedPlanning") {
            PriorityPlanner planner(instance);
            PlannerOptions options(planning_time_limit, 1000000);
            options.log_fname = benchmark_fname;
            bool success = planner.plan(options);
            if (!success) {
                ROS_WARN("Failed to plan");
                return false;
            }
            success &= planner.getPlan(solution);
        }
        else {
            move_group->setNamedTarget(pose_name);
            move_group->setPlannerId(planner_name);
            move_group->setPlanningTime(planning_time_limit);
            moveit::planning_interface::MoveGroupInterface::Plan plan;
            bool success = (move_group->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
            if (!success) {
                ROS_WARN("Failed to plan, try again");
                return motion_plan(pose_name, instance, tpg_config, recursion + 1);
            }
            auto toc = std::chrono::high_resolution_clock::now();
            t_plan = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count() * 1e-9;
            success = validateSolution(instance, plan.trajectory_);
            if (!success) {
                ROS_WARN("Planned moveit trajectory has collision, try again");
                return motion_plan(pose_name, instance, tpg_config, recursion + 1);
            }
            success = checkStartGoal(last_pose_name, pose_name, plan.trajectory_);
            if (!success) {
                ROS_WARN("Start pose %s and goal pose %s are not the same as expected", last_pose_name.c_str(), pose_name.c_str());
                return motion_plan(pose_name, instance, tpg_config, recursion + 1);
            }
            convertSolution(instance, plan.trajectory_, solution);        
        }

        /*
        Build the TPG
        */
        if (build_tpg) {
            tpg.reset();
            bool success = tpg.init(instance, solution, tpg_config);
            if (!success) {
                ROS_WARN("Failed to initialize TPG, try again");
                return motion_plan(pose_name, instance, tpg_config, recursion + 1);
            }
        }
        cur_solution = solution;
        return true;
    }

    bool set_start(const std::string &pose_name) {
        std::map<std::string, double> goal_joints = move_group->getNamedTargetValues(pose_name);
        // set the current joint to goal joint
        for (int i = 0; i < num_robots; i++) {
            RobotPose pose = instance_->initRobotPose(i);
            for (size_t j = 0; j < 7; j++) {
                current_joints[i*7+j] = goal_joints[joint_names[i*7+j]];
                pose.joint_values[j] = goal_joints[joint_names[i*7+j]];
            }
            instance_->moveRobot(i, pose);
            instance_->updateScene();
        }   

        // update the name
        last_pose_name = pose_name;
        return true;
    }

    bool test(const std::string &pose_name, tpg::TPGConfig &tpg_config) {
        bool success = true;
    
        int total_dof = 0;
        for (int i = 0; i < num_robots; i++) {
            total_dof += instance_->getRobotDOF(i);
        }
        /*
        Set the start and goal poses for planner instance
        */
        std::map<std::string, double> goal_joints = move_group->getNamedTargetValues(pose_name);
        for (int i = 0; i < num_robots; i++) {
            std::vector<double> start_pose(7, 0.0);
            std::vector<double> goal_pose(7, 0.0);
            for (size_t j = 0; j < 7; j++) {
                start_pose[j] = current_joints[i*7+j];
                goal_pose[j] = goal_joints[joint_names[i*7+j]];
            }
            instance_->setStartPose(i, start_pose);
            instance_->setGoalPose(i, goal_pose);
        }

        MRTrajectory init_solution;
        moveit_msgs::RobotTrajectory plan_traj;
        bool is_moveit_format = false;
        double flowtime, makespan;

        if (load_tpg) {
            std::string tpg_fname = tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".txt";
            success &= loadTPG(tpg_fname);
            if (!success) {
                ROS_ERROR("Failed to load TPG from file: %s", tpg_fname.c_str());
                return false;
            }
            plan_traj.joint_trajectory.joint_names = move_group->getVariableNames();
            tpg.setSyncJointTrajectory(plan_traj.joint_trajectory, flowtime, makespan);
            if (validateSolution(instance_, plan_traj) == false) {
                ROS_ERROR("Plan trajectory has collision");
                return false;
            }
            if (!checkStartGoal(last_pose_name, pose_name, plan_traj)) {
                ROS_ERROR("Start pose %s and goal pose %s are not the same as expected", last_pose_name.c_str(), pose_name.c_str());
                return false;
            }
            is_moveit_format = true;
        }
        else if (load_cbs) {
            std::string cbs_fname = tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".csv";
            moveit_msgs::RobotTrajectory traj;
            traj.joint_trajectory.joint_names = move_group->getVariableNames();
            success &= loadSolution(instance_, cbs_fname, traj);
            if (!success) {
                ROS_ERROR("Failed to load CBS solution from file: %s", cbs_fname.c_str());
                return false;
            }
            ROS_INFO("Loaded CBS csv from file: %s", cbs_fname.c_str());
            MRTrajectory speedup_traj;
            convertSolution(instance_, traj, speedup_traj, true);
            rediscretizeSolution(instance_, speedup_traj, init_solution, tpg_config.dt);

            success = validateSolution(instance_, init_solution);
            if (!success) {
                ROS_ERROR("CBS solution has collision");
                return false;
            }
            for (int i = 0; i < num_robots; i++) {
                flowtime += init_solution[i].times.back();
                makespan = std::max(makespan, init_solution[i].times.back());
            }
            if (tpg_shortcut) {
                tpg.reset();
                tpg_config.one_robust = false;
                success &= tpg.init(instance_, init_solution, tpg_config);
                if (!success) {
                    ROS_WARN("Failed to initialize TPG from cbs solution");
                    return false;
                }
            }
        }
        else {
            success &= motion_plan(pose_name, instance_, tpg_config);
            if (!success) {
                return false;
            }
            init_solution = cur_solution;
            
            if (build_tpg) {
                std::string tpg_fname = tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".txt";
                success &= saveTPG(tpg_fname);
                
                if (success) {
                    std::string sol_fname = tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".sol.csv";
                    plan_traj.joint_trajectory.joint_names.resize(total_dof);
                    tpg.setSyncJointTrajectory(plan_traj.joint_trajectory, flowtime, makespan);
                    success &= validateSolution(instance_, plan_traj);
                    if (!success) {
                        ROS_ERROR("Planned trajectory has collision");
                        return false;
                    }
                    if (!checkStartGoal(last_pose_name, pose_name, plan_traj)) {
                        ROS_ERROR("Start pose %s and goal pose %s are not the same as expected", last_pose_name.c_str(), pose_name.c_str());
                        return false;
                    }
                }
                else {
                    return false;
                }
            }
        }



        moveit_msgs::RobotTrajectory smoothed_traj;
        if (sc_options.auto_selector || sc_options.comp_shortcut || sc_options.prioritized_shortcut || sc_options.path_shortcut
                || sc_options.thompson_selector || sc_options.round_robin) {
            Shortcutter sc(instance_, sc_options);
            MRTrajectory solution;
            if (!is_moveit_format) {
                success &= sc.shortcutSolution(init_solution, solution);
                // saveSolution(instance_, init_solution, tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".init.csv");
            }
            else {
                success &= sc.shortcutSolution(plan_traj, solution);
                // success &= saveSolution(instance_, plan_traj, tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".csv");
            }
            // shortcutSolution(instance_, plan_traj, smoothed_traj, sc_options);
            // success &= saveSolution(instance_, solution, tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".smoothed.csv");
            //log("saved smoothed solution", LogLevel::INFO);
            
            // convertSolution(instance_, smoothed_traj, solution, true);
            // log(solution[0], LogLevel::INFO);
            if (benchmark) {
                double new_makespan = sc.calculate_makespan(solution);
                logProgressFileAppend(benchmark_fname, last_pose_name, pose_name, t_plan, makespan, new_makespan);
            }
            bool success = validateSolution(instance_, solution);
            if (!success) {
                ROS_ERROR("Shortcut solution has collision");
                return false;
            }
            if (!benchmark) {
                tpg.reset();
                if (!tpg.init(instance_, solution, tpg_config)) {
                    ROS_ERROR("Failed to initialize TPG from the shortcut solution");
                    //tpg.saveToDotFile(tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".dot");
                    return false;
                }
            }
        }
        else if (chomp) {
            moveit_msgs::RobotTrajectory plan_traj;
            plan_traj.joint_trajectory.joint_names = move_group->getVariableNames();
            double flowtime, makespan;
            tpg.setSyncJointTrajectory(plan_traj.joint_trajectory, flowtime, makespan);

            moveit::core::RobotModelConstPtr robot_model = move_group->getRobotModel();
            auto group_name = move_group->getName();
            auto tic = std::chrono::high_resolution_clock::now();
            success &= optimizeTrajectory(instance_, plan_traj, group_name, robot_model, nh, smoothed_traj);
            auto toc = std::chrono::high_resolution_clock::now();
            double t_optimize = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count() * 1e-9;
            if (!success) {
                ROS_ERROR("Failed to optimize trajectory");
                return false;
            }
            
            MRTrajectory solution;
            convertSolution(instance_, smoothed_traj, solution);
            if (benchmark) {
                Shortcutter sc(instance_, sc_options);
                double new_makespan = sc.calculate_makespan(solution);
                logProgressFileAppend(benchmark_fname, last_pose_name, pose_name, t_plan, makespan, new_makespan);
            }
            bool success = validateSolution(instance_, solution);
            if (!success) {
                ROS_ERROR("Shortcut solution has collision");
                return false;
            }
            if (!benchmark) {
                tpg.reset();
                if (!tpg.init(instance_, solution, tpg_config)) {
                    ROS_ERROR("Failed to initialize TPG from the shortcut solution");
                    //tpg.saveToDotFile(tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".dot");
                    return false;
                }
            }
        }
        else if (build_tpg || load_tpg) {
            // tpg shortcut
            //tpg.saveToDotFile(tpg_savedir + "/" + last_pose_name + "_" + pose_name + ".dot");
            success &= tpg.optimize(instance_, tpg_config);
            //tpg.saveToDotFile(tpg_savedir + "/" + last_pose_name + "_" + pose_name + "_optim.dot");
            if (benchmark) {
                tpg.saveStats(benchmark_fname, last_pose_name, pose_name);
            }
            if (!success) {
                ROS_ERROR("Failed to optimize TPG");
                return false;
            }
            double flowtime, makespan;
            smoothed_traj.joint_trajectory.joint_names = move_group->getVariableNames();
            tpg.setSyncJointTrajectory(smoothed_traj.joint_trajectory, flowtime, makespan);
            if (validateSolution(instance_, smoothed_traj) == false) {
                ROS_ERROR("Optimized solution has collision");
                return false;
            }
        }

        if (!benchmark) {
            if (async) {
                std::vector<ros::ServiceClient> clients;
                if (mfi) {
                    clients.push_back(nh.serviceClient<moveit_msgs::ExecuteKnownTrajectory>("/yk_destroyer/yk_execute_trajectory"));
                    clients.push_back(nh.serviceClient<moveit_msgs::ExecuteKnownTrajectory>("/yk_architect/yk_execute_trajectory"));
                }
                else {
                    for (auto group_name: group_names) {
                        clients.push_back(nh.serviceClient<moveit_msgs::ExecuteKnownTrajectory>(group_name + "/yk_execute_trajectory"));
                    }
                }
                
                tpg.moveit_mt_execute(joint_names_split, clients);
            } else {
                tpg.moveit_execute(instance_, move_group);
            }
        }
        return success;
    }


    void left_arm_joint_state_cb(const sensor_msgs::JointState::ConstPtr& msg) {
        for (size_t i = 0; i < 6; i++) {
            current_joints[i] = msg->position[i];
        }
        tpg.update_joint_states(msg->position, 0);
        left_arm_joint_state_received = true;
    }

    void right_arm_joint_state_cb(const sensor_msgs::JointState::ConstPtr& msg) {
        for (size_t i = 0; i < 6; i++) {
            current_joints[7+i] = msg->position[i];
        }
        tpg.update_joint_states(msg->position, 1);
        right_arm_joint_state_received = true;
    }

    void all_arm_joint_state_cb(const sensor_msgs::JointState::ConstPtr& msg) {
        current_joints = msg->position;
        for (size_t i = 0; i < instance_->getNumberOfRobots(); i++) {
            std::vector<double> joints(7, 0.0);
            for (size_t j = 0; j < 7; j++) {
                joints[j] = msg->position[i*7+j];
            }
            tpg.update_joint_states(joints, i);
        }
        left_arm_joint_state_received = true;
        right_arm_joint_state_received = true;
    }

    void set_load_tpg(bool load_tpg) {
        this->load_tpg = load_tpg;
    }

    void set_load_cbs(bool load_cbs) {
        this->load_cbs = load_cbs;
    }

    bool checkStartGoal(const std::string &start_name, const std::string &goal_name, const moveit_msgs::RobotTrajectory &traj) {
        /*
        Check if the start and goal poses are correct 
        */
        
        std::map<std::string, double> start_joints = move_group->getNamedTargetValues(start_name);
        std::map<std::string, double> goal_joints = move_group->getNamedTargetValues(goal_name);
        auto start_point = traj.joint_trajectory.points.front().positions;
        auto goal_point = traj.joint_trajectory.points.back().positions;
        for (int i = 0; i < joint_names.size(); i++) {
            if (start_name == "default") {
                start_joints[joint_names[i]] = current_joints[i];
            }
            // check if they are the same angle (within 1e-3 or a full 2\pi difference)
            double diff = start_joints[joint_names[i]] - start_point[i];
            diff = fmod(fmod(diff + M_PI, 2*M_PI) + 2*M_PI, 2 * M_PI) - M_PI;
        
            if (fabs(diff) > 0.02) {
                ROS_ERROR("Start point %f does not match the start pose %s %f", start_point[i], joint_names[i].c_str(), start_joints[joint_names[i]]);
                return false;
            }

            diff = goal_joints[joint_names[i]] - goal_point[i];
            diff = fmod(fmod(diff + M_PI, 2*M_PI) + 2*M_PI, 2 * M_PI) - M_PI;

            if (fabs(diff) > 0.02) {
                ROS_ERROR("Goal point %f does not match the goal pose %s %f", goal_point[i], joint_names[i].c_str(), goal_joints[joint_names[i]]);
                return false;
            }

        }
        return true;
    }

private:
    ros::NodeHandle nh;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group;
    planning_scene::PlanningScenePtr planning_scene;
    robot_state::RobotStatePtr kinematic_state;
    std::shared_ptr<MoveitInstance> instance_;
    ros::Subscriber left_arm_sub, right_arm_sub, all_arm_sub;
    ros::ServiceClient planning_scene_diff_client;

    int num_robots;
    std::string pose_name;
    std::string last_pose_name="default";
    std::string planner_name;
    std::string benchmark_fname;
    std::string tpg_savedir;
    std::vector<std::string> group_names;
    std::vector<std::string> eof_groups;
    std::vector<std::string> joint_names;
    std::vector<std::vector<std::string>> joint_names_split;
    std::vector<double> current_joints;

    tpg::TPG tpg;
    bool left_arm_joint_state_received = false;
    bool right_arm_joint_state_received = false;
    bool async;
    bool mfi;
    bool load_tpg;
    bool load_cbs;
    bool build_tpg;
    bool benchmark;
    ShortcutOptions sc_options;
    double planning_time_limit = 2.0;
    double tpg_time_limit = 2.0;
    bool chomp;
    bool tpg_shortcut;
    double t_plan = 0.0;
    MRTrajectory cur_solution;
};
    

int main(int argc, char** argv) {
    ros::init(argc, argv, "plan_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // read parameters
    std::string loglevel = "info";
    if (nh_private.hasParam("loglevel")) {
        nh_private.getParam("loglevel", loglevel);
    }
    setLogLevel(loglevel);
    
    int num_robots = 2;
    bool async = true;
    bool mfi = true;
    bool shortcut = true;
    bool benchmark = true;
    bool load_tpg = false;
    bool load_cbs = false;
    bool build_tpg = false;
    bool chomp = false;
    double planning_time_limit = 2.0;
    std::string pose_name = "left_push_up";
    std::string movegroup_name = "dual_arms";
    std::string planner_name = "PrioritizedPlanning";
    std::string benchmark_fname = "stats.csv";
    std::string tpg_savedir = "outputs/tpgs";
    std::vector<std::string> group_names = {"left_arm", "right_arm"};

    if (nh_private.hasParam("num_robots")) {
        nh_private.getParam("num_robots", num_robots);
    }
    group_names.resize(num_robots);
    for (int i = 0; i < num_robots; i++) {
        if (nh_private.hasParam("group_name_" + std::to_string(i))) {
            nh_private.getParam("group_name_" + std::to_string(i), group_names[i]);
        }
    }
    std::vector<std::string> eof_groups;
    for (int i = 0; i < num_robots; i++) {
        if (nh_private.hasParam("eof_group_" + std::to_string(i))) {
            std::string eof_group;
            nh_private.getParam("eof_group_" + std::to_string(i), eof_group);
            eof_groups.push_back(eof_group);
        }
        else {
            ROS_WARN("End effector group for robot %d not found", i);
        }
    }

    if (nh_private.hasParam("mfi")) {
        nh_private.getParam("mfi", mfi);
    }
    if (nh_private.hasParam("async")) {
        nh_private.getParam("async", async);
    }
    if (nh_private.hasParam("pose_name")) {
        nh_private.getParam("pose_name", pose_name);
    }
    if (nh_private.hasParam("planner_name")) {
        nh_private.getParam("planner_name", planner_name);
    }
    if (nh_private.hasParam("planning_time_limit")) {
        nh_private.getParam("planning_time_limit", planning_time_limit);
    }
    if (nh_private.hasParam("shortcut")) {
        nh_private.getParam("shortcut", shortcut);
    }
    if (nh_private.hasParam("benchmark")) {
        nh_private.getParam("benchmark", benchmark);
    }
    if (nh_private.hasParam("movegroup_name")) {
        nh_private.getParam("movegroup_name", movegroup_name);
    }
    if (nh_private.hasParam("output_file")) {
        nh_private.getParam("output_file", benchmark_fname);
    }
    if (nh_private.hasParam("tpg_savedir")) {
        nh_private.getParam("tpg_savedir", tpg_savedir);
    }
    if (nh_private.hasParam("load_tpg")) {
        nh_private.getParam("load_tpg", load_tpg);
    }
    if (nh_private.hasParam("load_cbs")) {
        nh_private.getParam("load_cbs", load_cbs);
    }
    if (nh_private.hasParam("build_tpg")) {
        nh_private.getParam("build_tpg", build_tpg);
    }
    if (!benchmark) {
        build_tpg = true;
    }

    // wait 100 miliseconds for the move_group_interface to be ready
    ros::Duration(0.1).sleep();
    // Declare the MoveGroupInterface
    auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(movegroup_name);

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    
    // Create the planning scene from robot model
    // Planning scene monitor.
    planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>("robot_description");
    planning_scene_monitor->startSceneMonitor();
    planning_scene_monitor->startStateMonitor();
    planning_scene_monitor->startWorldGeometryMonitor();
    planning_scene_monitor->requestPlanningSceneState();
    // wait 300 miliseconds for the planning scene to be ready
    ros::Duration(0.3).sleep();
    planning_scene::PlanningScenePtr planning_scene = planning_scene_monitor->getPlanningScene();

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    auto robot_model = robot_model_loader.getModel();
    auto kinematic_state = std::make_shared<robot_state::RobotState>(robot_model);
    kinematic_state->setToDefaultValues();


    ShortcutOptions sc_options;
    tpg::TPGConfig tpg_config;
    tpg_config.random_shortcut = true;
    tpg_config.shortcut_time = 0.1;
    tpg_config.shortcut = shortcut;
    tpg_config.tight_shortcut = true;
    tpg_config.forward_doubleloop = false;
    tpg_config.backward_doubleloop = false;
    tpg_config.forward_singleloop = true;
    tpg_config.biased_sample = false;
    tpg_config.subset_shortcut = false;
    tpg_config.subset_prob = 0.4;
    tpg_config.progress_file = benchmark_fname;
    tpg_config.log_interval = 1.0;
    tpg_config.comp_shortcut = false;
    tpg_config.dt = 0.1;
    if (nh_private.hasParam("shortcut_time")) {
        nh_private.getParam("shortcut_time", tpg_config.shortcut_time);
    }
    if (nh_private.hasParam("random_shortcut")) {
        nh_private.getParam("random_shortcut", tpg_config.random_shortcut);
    }
    if (nh_private.hasParam("tight_shortcut")) {
        nh_private.getParam("tight_shortcut", tpg_config.tight_shortcut);
    }
    if (nh_private.hasParam("forward_doubleloop")) {
        nh_private.getParam("forward_doubleloop", tpg_config.forward_doubleloop);
    }
    if (nh_private.hasParam("backward_doubleloop")) {
        nh_private.getParam("backward_doubleloop", tpg_config.backward_doubleloop);
    }
    if (nh_private.hasParam("forward_singleloop")) {
        nh_private.getParam("forward_singleloop", tpg_config.forward_singleloop);
    }
    if (nh_private.hasParam("biased_sample")) {
        nh_private.getParam("biased_sample", tpg_config.biased_sample);
    }
    if (nh_private.hasParam("subset_shortcut")) {
        nh_private.getParam("subset_shortcut", tpg_config.subset_shortcut);
    }
    if (nh_private.hasParam("subset_prob")) {
        nh_private.getParam("subset_prob", tpg_config.subset_prob);
    }
    if (nh_private.hasParam("log_interval")) {
        nh_private.getParam("log_interval", tpg_config.log_interval);
    }
    if (nh_private.hasParam("composite_shortcut")) {
        nh_private.getParam("composite_shortcut", tpg_config.comp_shortcut);
    }
    if (nh_private.hasParam("prioritized_shortcut")) {
        nh_private.getParam("prioritized_shortcut", tpg_config.allow_col);
    }
    if (nh_private.hasParam("check_dt")) {
        nh_private.getParam("check_dt", tpg_config.dt);
    }

    bool tpg_shortcut = false;
    if (nh_private.hasParam("tpg_shortcut")) {
        nh_private.getParam("tpg_shortcut", tpg_shortcut);
    }
    if (nh_private.hasParam("path_shortcut")) {
        nh_private.getParam("path_shortcut", sc_options.path_shortcut);
    }
    if (nh_private.hasParam("auto_selector")) {
        nh_private.getParam("auto_selector", sc_options.auto_selector);
    }
    if (nh_private.hasParam("thompson_selector")) {
        nh_private.getParam("thompson_selector", sc_options.thompson_selector);
    }
    if (nh_private.hasParam("round_robin")) {
        nh_private.getParam("round_robin", sc_options.round_robin);
    }
    if (!tpg_shortcut) {
        sc_options.comp_shortcut = tpg_config.comp_shortcut;
        sc_options.prioritized_shortcut = tpg_config.allow_col;
        sc_options.forward_doubleloop = tpg_config.forward_doubleloop;
        sc_options.backward_doubleloop = tpg_config.backward_doubleloop;
        sc_options.forward_singleloop = tpg_config.forward_singleloop;
    }

    sc_options.dt = tpg_config.dt;
    sc_options.t_limit = tpg_config.shortcut_time;
    sc_options.log_interval = tpg_config.log_interval;
    sc_options.progress_file = tpg_config.progress_file;

    auto pp_tester = TestPPPlanning(nh, move_group, planning_scene, kinematic_state, group_names, eof_groups,
        planner_name, planning_time_limit, async, mfi, load_tpg, benchmark, load_cbs, chomp, tpg_shortcut, build_tpg,
        sc_options, benchmark_fname, tpg_savedir);
    pp_tester.setup_once();

    std::vector<std::string> pose_names = {pose_name};
    for (int i=1; ; i++ ) {
        if (!nh_private.hasParam("pose_name" + std::to_string(i))) {
            break;
        }
        nh_private.getParam("pose_name" + std::to_string(i), pose_name);
        pose_names.push_back(pose_name);
    }

    if (benchmark) {
        int failed = 0;
        int total = pose_names.size() * (pose_names.size() - 1);
        for (int i = 0; i < pose_names.size(); i++) {
            for (int j = 0 ; j < pose_names.size(); j++) {
                if (i != j) {
                    bool success = pp_tester.set_start(pose_names[i]);
                    success &= pp_tester.test(pose_names[j], tpg_config);
                    if (!success) {
                        ROS_ERROR("%s Failed to plan from pose %s to pose: %s", movegroup_name.c_str(), pose_names[i].c_str(), pose_names[j].c_str());
                        failed ++;
                    }
                }
            }
        }
        ROS_INFO("%s Failed %d / %d times", movegroup_name.c_str(), failed, total);
    }
    else {
        bool success = true;
        pp_tester.set_load_tpg(false);
        pp_tester.set_load_cbs(false);
        pp_tester.test(pose_names[0], tpg_config);
        pp_tester.set_load_tpg(load_tpg);
        pp_tester.set_load_cbs(load_cbs);
        pp_tester.set_start(pose_names[0]);
        for (int i = 1; i < pose_names.size(); i++) {
            success &= pp_tester.test(pose_names[i], tpg_config);
            pp_tester.set_start(pose_names[i]);
            if (!success) {
                ROS_ERROR("%s Failed to plan for pose: %s", movegroup_name.c_str(), pose_names[i].c_str());
                ros::shutdown();
                return -1;
            }
        }

    }
    
    ROS_INFO("Planning completed successfully");
    ros::shutdown();
    return 0;
}
