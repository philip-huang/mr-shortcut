import subprocess
import time
import os
import multiprocessing as mp
import signal
import sys
from typing import List
from multiprocessing import Process

class ProcessManager:
    def __init__(self):
        self.active_processes: List[Process] = []
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def add_processes(self, processes: List[Process]):
        self.active_processes.extend(processes)

    def cleanup(self):
        print("\nCleaning up processes...")
        for p in self.active_processes:
            if p.is_alive():
                print(f"Terminating process {p.pid}")
                p.terminate()
                p.join(timeout=3)
                
                if p.is_alive():
                    print(f"Force killing process {p.pid}")
                    p.kill()
                    p.join()
        print("All processes cleaned up")
        # Clear the list after cleanup
        self.active_processes.clear()

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}")
        self.cleanup()
        sys.exit(0)

    def wait_for_processes(self):
        try:
            for p in self.active_processes:
                p.join()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received")
            self.cleanup()
            sys.exit(0)


def run_roslaunch(package, launch_file, params):
    # Start roslaunch
    roslaunch = subprocess.Popen(['roslaunch', package, launch_file, *['{}:={}'.format(k, v) for k, v in params.items()]])

    # Wait for roslaunch to finish
    roslaunch.wait()

def run_script(script_path, params):
    # Run another script
    script = subprocess.Popen(['python3', script_path, *['--{}={}'.format(k, v) for k, v in params.items()]])
    script.wait()

def eval_setting(ns, robot_name, load_tpg, load_cbs, t, tight, biased, partial, subset_prob, planner_name, planning_time_limit,
                 loop_type):
    assert loop_type in ['fwd_diter', 'bwd_diter', 'iter', 'comp', 'pp', 'random', 'comp_random', 'pp_random', 'path', 'auto', 'rr', 'thompson']

    base_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../outputs/')
    tpg_directory = None
    if load_tpg:
        directory = base_directory + \
            f'{loop_type}_{("tight" if tight else "loose")}' + \
            f'{"_biased" if biased else ""}{f"_partial{subset_prob}" if partial else ""}'
        tpg_directory = base_directory + f'tpgs/t={planning_time_limit}_{planner_name}_{robot_name}'
    elif load_cbs:
        directory = base_directory + f'cbs_{loop_type}_{("tight" if tight else "loose")}' + \
                    f'{"_biased" if biased else ""}{f"_partial{subset_prob}" if partial else ""}'
        tpg_directory = base_directory + f'cbs/{robot_name}'
    else:
        directory = base_directory + f't={planning_time_limit}_{planner_name}'
        tpg_directory = base_directory + f'tpgs/t={planning_time_limit}_{planner_name}_{robot_name}'

    if not os.path.exists(directory):
        # If not, create the directory
        os.makedirs(directory)

    # Set parameters
    params = {
        'benchmark': 'true',
        'use_rviz': 'false',
        'random_shortcut': 'true' if (loop_type in ['random', 'comp_random', 'pp_random']) else 'false',
        'shortcut_time': str(t),
        'tight_shortcut': 'true' if tight else 'false',
        'planner_name': planner_name,
        'planning_time_limit': planning_time_limit,
        'ns': ns,
        'output_file': f'{directory}/{robot_name}_benchmark.csv', 
        'load_tpg': 'true' if load_tpg else 'false',
        'load_cbs': 'true' if load_cbs else 'false',
        'tpg_shortcut': 'true' if (loop_type in ['random', 'comp_random', 'pp_random']) else 'false',
        'prioritized_shortcut': 'true' if (loop_type in ['pp', 'pp_random']) else 'false',
        'composite_shortcut': 'true' if (loop_type in ['comp', 'comp_random']) else 'false',
        'path_shortcut': 'true' if loop_type in ['path', 'fwd_diter', 'bwd_diter', 'iter'] else 'false',
        'auto_selector': 'true' if loop_type == 'auto' else 'false',
        'round_robin': 'true' if loop_type == 'rr' else 'false',
        'thompson_selector': 'true' if loop_type == 'thompson' else 'false',
        'tpg_savedir': tpg_directory,
        'forward_doubleloop': 'true' if loop_type == 'fwd_diter' else 'false',
        'backward_doubleloop': 'true' if loop_type == 'bwd_diter' else 'false',
        'forward_singleloop': 'true' if loop_type == 'iter' else 'false',
        'biased_sample': 'true' if biased else 'false',
        'subset_shortcut': 'true' if partial else 'false',
        # Add more parameters as needed
    }

    # Run roslaunch
    print(params)
    run_roslaunch('mr-shortcut', f'{robot_name}.launch', params)

    # Wait for a while to make sure everything has started up
    time.sleep(2)

    if load_tpg:
        script_params = {
            'input': f'{directory}/{robot_name}_benchmark.csv',
            'output': f'{directory}/{robot_name}_benchmark.avg.csv',
        }
        # Run another script
        run_script('count_stats.py', script_params)


# run the evaluations in parallel
def add_planner_processes(envs, times, id = 0):
    processes = []

    biased = False
    shortcut_time = 0.0
    load_tpg = False
    load_cbs = False
    tight = False
    partial = False
    subset_prob = 0.4
    planner_name = 'RRTConnect'
    for env in envs:
        for planning_time in times:
            ns = f'run_{id}'
            id += 1
            p = mp.Process(target=eval_setting, 
                            args=(ns, env, load_tpg, load_cbs, shortcut_time, tight, biased, partial, subset_prob, 
                                  planner_name, planning_time, 'iter'))
            p.start()
            processes.append(p)
            time.sleep(1)
         
    return processes, id

def add_shortcut_processes(envs, shortcut_ts, load_types, loop_types, id = 0):
    processes = []

    planning_time = 15.0
    planner_name = 'RRTConnect'
    subset_prob = 0.6
    for env, shortcut_t in zip(envs, shortcut_ts):
        for load_type in load_types:
            load_tpg = True if load_type == 'tpg' else False
            load_cbs = True if load_type == 'cbs' else False
            for tight in [False]:
                for biased in [False]:
                    for partial in [False]:
                        for loop_type in loop_types:
                            ns = f'run_{id}'
                            id += 1
                            p = mp.Process(target=eval_setting, 
                                            args=(ns, env, load_tpg, load_cbs, shortcut_t, tight, biased, partial, subset_prob, 
                                                planner_name, planning_time, loop_type))
                            p.start()
                            processes.append(p)
                            time.sleep(1)
    return processes, id


if __name__ == "__main__":
    # Initialize process manager
    process_manager = ProcessManager()

    envs = ["dual_gp4", "panda_two", "panda_two_rod", "panda_four", "panda_three", "panda_four_bins"]
    t_plans = [0.5, 0.5, 2.0, 15.0, 15.0, 15.0]
    shortcut_ts = [5.0, 5.0, 10.0, 60.0, 60.0, 60.0]
    loop_types = ['path', 'pp', 'rr', 'auto', 'thompson']

    for env, t in zip(envs, t_plans):
        processes, id = add_planner_processes([env], [t])
        process_manager.add_processes(processes)
        process_manager.wait_for_processes()

    #id = 1
    #processes, id = add_shortcut_processes(envs[1:2], shortcut_ts[1:2], ['tpg'], ['iter', 'bwd_diter'], id=id)
    #process_manager.add_processes(processes)
    #process_manager.wait_for_processes()

    # for i in range(2, 3):
    #     processes, id = add_shortcut_processes(envs[i:i+1], shortcut_ts[i:i+1], ['cbs'], loop_types, id=id)
    #     process_manager.add_processes(processes)
    #     process_manager.wait_for_processes()
    # loop_types = ['fwd_diter', 'bwd_diter', 'iter', 'random', 'comp']
    # for i in range(2, 3):
    #     processes, id = add_shortcut_processes(envs[i:i+1], shortcut_ts[i:i+1], ['cbs'], loop_types, id=id)
    #     process_manager.add_processes(processes)
    #     process_manager.wait_for_processes()   
        
