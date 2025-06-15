#ifndef MR_SHORTCUT_PLANNER_H
#define MR_SHORTCUT_PLANNER_H

#include "instance.h" // Include the abstract problem instance definition
#include "SingleAgentPlanner.h"
#include <memory>
#include <vector>

// Abstract planner class
class AbstractPlanner {
public:
    // Initialize the planner with a specific planning problem instance
    AbstractPlanner(std::shared_ptr<PlanInstance> instance) : instance_(instance) {
        num_robots_ = instance->getNumberOfRobots();
    }
    
    // Perform the planning process
    virtual bool plan(const PlannerOptions &options) = 0;

    // Retrieve the plan (if needed, depending on your design, this could return a path, a series of actions, etc.)
    // For simplicity, this could return a boolean indicating success for now,
    // but you might want to define a more complex structure for the plan itself.
    virtual bool getPlan(MRTrajectory &solution) const = 0;

    virtual ~AbstractPlanner() = default;

    double getPlanTime() const {
        return planning_time_;
    }

protected:
    int num_robots_;
    std::shared_ptr<PlanInstance> instance_;
    double planning_time_ = 0;
};



// Example of a concrete planner class that implements the AbstractPlanner interface
// This is where you would implement specific planning algorithms
class PriorityPlanner : public AbstractPlanner {
public:
    PriorityPlanner(std::shared_ptr<PlanInstance> instance);

    virtual bool plan(const PlannerOptions &options) override;

    virtual bool getPlan(MRTrajectory &solution) const override;

protected:
    std::vector<SingleAgentPlannerPtr> agent_planners_;
    MRTrajectory solution_;
    bool solved = false;
};

// utils
bool convertSolution(std::shared_ptr<PlanInstance> instance,
                    const moveit_msgs::RobotTrajectory &plan_traj,
                    MRTrajectory &solution,
                    bool reset_speed = true);


bool convertSolution(std::shared_ptr<PlanInstance> instance,
                    const moveit_msgs::RobotTrajectory &plan_traj,
                    int robot_id,
                    RobotTrajectory &solution);

bool saveSolution(std::shared_ptr<PlanInstance> instance,
                  const moveit_msgs::RobotTrajectory &plan_traj,
                  const std::string &file_name);
                  
bool saveSolution(std::shared_ptr<PlanInstance> instance,
                  const MRTrajectory &synced_traj,
                  const std::string &file_name);

/* time is assumed to be uniform as dt */
bool loadSolution(std::shared_ptr<PlanInstance> instance,
                  const std::string &file_name,
                  double dt,
                  moveit_msgs::RobotTrajectory &plan_traj);

/* time is supplied in the first column*/
bool loadSolution(std::shared_ptr<PlanInstance> instance,
                  const std::string &file_name,
                  moveit_msgs::RobotTrajectory &plan_traj);

bool validateSolution(std::shared_ptr<PlanInstance> instance,
                    const MRTrajectory &solution,
                    double col_dt);

/* assuming uniform discretiziation, check for collisions*/
bool validateSolution(std::shared_ptr<PlanInstance> instance,
                       const MRTrajectory &solution);

void retimeSolution(std::shared_ptr<PlanInstance> instance,
                    const MRTrajectory &solution,
                    MRTrajectory &retime_solution,
                    double dt);

void rediscretizeSolution(std::shared_ptr<PlanInstance> instance,
                    const moveit_msgs::RobotTrajectory &plan_traj,
                    moveit_msgs::RobotTrajectory &retime_traj,
                    double new_dt);

void rediscretizeSolution(std::shared_ptr<PlanInstance> instance,
                        const MRTrajectory &solution,
                        MRTrajectory &retime_solution,
                        double new_dt);
void removeWait(std::shared_ptr<PlanInstance> instance,
                        MRTrajectory &solution);
bool validateSolution(std::shared_ptr<PlanInstance> instance,
                     const moveit_msgs::RobotTrajectory &plan_traj);

bool optimizeTrajectory(std::shared_ptr<PlanInstance> instance,
                        const moveit_msgs::RobotTrajectory& input_trajectory,
                        const std::string& group_name,
                        robot_model::RobotModelConstPtr robot_model,
                        const ros::NodeHandle& node_handle,
                        moveit_msgs::RobotTrajectory& smoothed_traj
                        );
struct SmoothnessMetrics {
    double normalized_jerk_score;
    double directional_consistency;
};
SmoothnessMetrics calculate_smoothness(const MRTrajectory &synced_plan, std::shared_ptr<PlanInstance> instance);

#endif // MR_SHORTCUT_PLANNER_H
