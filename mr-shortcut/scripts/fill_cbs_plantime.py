import pandas as pd
import os.path as osp

def fill_planning_time(benchmark_file, mramp_output_file, output_file):
    """
    Fills in the 't_plan' column in the benchmark CSV file using data from the output CSV file.

    Args:
        benchmark_file: Path to the benchmark CSV file (e.g., 'dual_gp4_benchmark.csv').
        output_file: Path to the output CSV file (e.g., 'mramp_output.csv').
    """

    try:
        # Load the benchmark and output data into pandas DataFrames
        df_benchmark = pd.read_csv(benchmark_file)
        df_output = pd.read_csv(mramp_output_file)
    except FileNotFoundError:
        print(f"Error: One or both of the CSV files were not found.")
        return

    # Create a dictionary to store planning times from the output file
    planning_times = {}
    for index, row in df_output.iterrows():
        description = row['description']
        planning_time = row['planning_time']

        # Extract start and goal poses from the description
        if "from" in description and "to" in description:
          parts = description.split()
          start_pose = parts[parts.index("from") + 1]
          goal_pose = parts[parts.index("to") + 1]

          # Store the planning time in the dictionary
          planning_times[(start_pose, goal_pose)] = planning_time
        
    

    # Fill in the 't_plan' column in the benchmark DataFrame
    for index, row in df_benchmark.iterrows():
        start_pose = row['start_pose']
        goal_pose = row['goal_pose']

        if (start_pose, goal_pose) in planning_times:
            df_benchmark.loc[index, 't_plan'] = planning_times[(start_pose, goal_pose)]
        else:
            print(f"Warning: No matching planning time found for {start_pose} to {goal_pose}")

    # Save the updated benchmark DataFrame to a new CSV file
    df_benchmark.to_csv(output_csv, index=False)
    print("Updated benchmark file saved to 'updated_dual_gp4_benchmark.csv'")

# Example usage:
folder = '../outputs/cbs'
env = 'panda_three'
benchmark_csv = osp.join(folder, f'{env}_benchmark.csv')
mramp_output_csv = osp.join(folder, env, 'mramp_output.csv')
output_csv = osp.join(folder, f'{env}_benchmark.csv')

fill_planning_time(benchmark_csv, mramp_output_csv, output_csv)
