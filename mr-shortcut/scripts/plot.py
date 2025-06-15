import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from count_stats import average_improvement
import os

# Read the data from the csv file
def read_csv(dir, env):
    file1 = os.path.join(dir, f'{env}_benchmark.csv')
    df = pd.read_csv(file1, sep=',')
    df[['start_pose', 'goal_pose']] = df[['start_pose', 'goal_pose']].bfill()
    df['timestep'] = df.groupby(['start_pose', 'goal_pose']).cumcount() + 1

    # count the number of unique (start_pose, goal_pose) pairs
    print(f"{dir}/{env} {len(df.groupby(['start_pose', 'goal_pose']).size())}")

    time_steps_count = df.groupby(['start_pose', 'goal_pose']).size().reset_index(name='time_steps_count')
    min_time_steps = time_steps_count['time_steps_count'].min()

    # Create _pre columns by propagating first values within each group
    first_values = df.groupby(['start_pose', 'goal_pose']).first()[['dir_consistency', 'pathlen']]
    first_values.columns = ['dir_consistency_pre', 'pathlen_pre']
    
    # Merge these values back to every row in the group
    df = df.join(first_values, on=['start_pose', 'goal_pose'])
    df = df.groupby(['start_pose', 'goal_pose']).head(min_time_steps).reset_index(drop=True)

    df['flowtime_improv'] = (df['flowtime_pre'] - df['flowtime_post']) / df['flowtime_pre'] * 100
    df['makespan_improv'] = (df['makespan_pre'] - df['makespan_post']) / df['makespan_pre'] * 100

    df['pathlen_improv'] = (df['pathlen_pre'] - df['pathlen']) / df['pathlen_pre'].clip(lower=1e-10) * 100
    df['dir_consistency_improv'] = (df['dir_consistency_pre'] - df['dir_consistency']) / df['dir_consistency_pre'].clip(lower=1e-10) * 100

    df['flowtime_diff_per_step'] = (df['flowtime_pre'] - df['flowtime_post']) / df['n_valid']
    df['makespan_diff_per_step'] = (df['makespan_pre'] - df['makespan_post']) / df['n_valid']
    
    df['flowtime_diff_per_step'] = df['flowtime_diff_per_step'].replace(-np.inf, np.nan).clip(lower=0)
    df['makespan_diff_per_step'] = df['makespan_diff_per_step'].replace(-np.inf, np.nan).clip(lower=0)
    return df

# Function to compute average statistics
def compute_statistics(group):
    columns_to_compute = [
        'flowtime_pre', 'makespan_pre', 'flowtime_post', 'makespan_post',
        't_init', 't_shortcut', 't_mcp', 't_check', 'n_check', 'n_valid',
        'makespan_improv', 'flowtime_improv', 'makespan_diff_per_step', 'n_colcheck_post',
        'pathlen', 'pathlen_pre', 'dir_consistency_pre',
        'norm_isj', 'dir_consistency', 'pathlen_improv', 'dir_consistency_improv',
        'n_comp', 'n_path', 'n_pp',
        'n_v_comp', 'n_v_path', 'n_v_pp'
    ]
    
    stats = {}
    
    for col in columns_to_compute:
        if col in group:
            stats[f'{col}_mean'] = group[col].mean()
            stats[f'{col}_std'] = group[col].std()
            #stats[f'{col}_median'] = group[col].median()
    
    return pd.Series(stats)

# Plot the data
def plot(env, folder="outputs_pikachu", planner=""):
    dt_dict = {'dual_gp4': 0.025,
               'panda_two': 0.05,
               'panda_two_rod': 0.05,
               'panda_four': 1.0,
               'panda_four_bins': 1.0,
               'panda_three': 1.0}
    dt = dt_dict[env]
    entries = [
                ('g', 'x', f'{planner}comp_loose', 'Composite'),
                ('purple', 'D', f'{planner}random_loose', 'TPG'),
                ('orange', 'P', f'{planner}pp_loose', 'Prioritized'), 
                #('b', 'o', f'{planner}pp_random_loose', 'TPG with Allow Collision'),
                ('cyan', '.', f'{planner}path_loose', 'Path'),
                ('r', 's', f'{planner}auto_loose', 'Weighted Discrete'),
                ('m', '^', f'{planner}rr_loose', 'RR'),
                ('pink', 'D', f'{planner}thompson_loose', 'Thompson'),
                ('black', 'v', f'{planner}fwd_diter_loose', 'Forward Double'),
                ('brown', '>', f'{planner}bwd_diter_loose', 'Backward Double'),
                ('yellow', 'o', f'{planner}iter_loose', 'Iterative'),
            ]
    metric = 'makespan'

    plt.figure(figsize=(16, 9))
    plt.title(f'{env}')
    plt.rcParams.update({'font.size': 15})

    base_dir = os.path.dirname(os.path.realpath(__file__))

    for color, marker, algo, label in entries:

        # get cur file dir
        dir = os.path.join(base_dir, f'../{folder}/{algo}')
        df = read_csv(dir, env)

        # Group by start_pose and goal_pose, then apply the function
        grouped_stats = df.groupby('timestep').apply(compute_statistics).reset_index()
        t_val_actual = grouped_stats['timestep'] * dt
        
        # print the number of unique (start_pose, goal_pose) pairs
        print(algo, len(df.groupby(['start_pose', 'goal_pose']).size()))
        
        # Plot the data with the specified color and label
        plt.subplot(3, 2, 1)
        
        vals = grouped_stats['makespan_improv_mean'][:-1]
        stds = grouped_stats['makespan_improv_std'][:-1]
        plt.plot(t_val_actual[:-1], vals, label=label, marker=marker, color=color)
        plt.fill_between(t_val_actual[:-1], np.array(vals) - np.array(stds), np.array(vals) + np.array(stds), alpha=0.2, color=color)
        
        plt.subplot(3, 2, 2)
        # split array of tuple into two arrays
        n_check = grouped_stats['n_check_mean'][1:-1]
        stds = grouped_stats['n_check_std'][1:-1]
        plt.plot(t_val_actual[1:-1], n_check, label=label, marker=marker, color=color)
        plt.fill_between(t_val_actual[1:-1], np.array(n_check) - np.array(stds), np.array(n_check) + np.array(stds), alpha=0.2, color=color)

        plt.subplot(3, 2, 3)
        n_valid = grouped_stats['n_valid_mean'][1:-1]
        stds = grouped_stats['n_valid_std'][1:-1]
        plt.plot(t_val_actual[1:-1], n_valid, label=label, marker=marker, color=color)
        plt.fill_between(t_val_actual[1:-1], np.array(n_valid) - np.array(stds), np.array(n_valid) + np.array(stds), alpha=0.2, color=color)

        plt.subplot(3, 2, 4)
        vals = grouped_stats['n_colcheck_post_mean'][1:-1]
        stds = grouped_stats['n_colcheck_post_std'][1:-1]
        plt.plot(t_val_actual[1:-1], vals, label=label, marker=marker, color=color)
        plt.fill_between(t_val_actual[1:-1], np.array(vals) - np.array(stds), np.array(vals) + np.array(stds), alpha=0.2, color=color)

        plt.subplot(3, 2, 5)
        vals = grouped_stats['norm_isj_mean'][1:-1]
        stds = grouped_stats['norm_isj_std'][1:-1]
        plt.plot(t_val_actual[1:-1], vals, label=label, marker=marker, color=color)
        plt.fill_between(t_val_actual[1:-1], np.array(vals) - np.array(stds), np.array(vals) + np.array(stds), alpha=0.2, color=color)

        plt.subplot(3, 2, 6)
        vals = grouped_stats['dir_consistency_mean'][1:-1]
        stds = grouped_stats['dir_consistency_std'][1:-1]
        plt.plot(t_val_actual[1:-1], vals, label=label, marker=marker, color=color)
        plt.fill_between(t_val_actual[1:-1], np.array(vals) - np.array(stds), np.array(vals) + np.array(stds), alpha=0.2, color=color)

    plt.subplot(3, 2, 1) 

    # log axis for x
    # lower bound the y axis to 0
    #plt.ylim(bottom=0)
    plt.ylabel('Improvement (%)')
    plt.xscale('log')

    plt.legend(fontsize='8')

    plt.subplot(3, 2, 2)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('# Shortcut Checked')

    plt.subplot(3, 2, 3)
    plt.ylabel('# Shortcut Valid')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(3, 2, 4)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('# Collision Check')

    plt.subplot(3, 2, 5)
    plt.xscale('log')
    plt.ylabel('Norm ISJ')

    plt.subplot(3, 2, 6)
    plt.xscale('log')
    plt.ylabel('Dir Consistency')

    plt.subplot(3, 2, 7)
    plt.xlabel('Time (s)')
    plt.xscale('log')
    plt.ylabel('% comp')

    plt.subplot(3, 2, 8)
    plt.xlabel('Time (s)')
    plt.xscale('log')
    plt.ylabel('% path')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'../outputs/plots/{env}.png'))
    plt.show()

def plot_comparison(metric='makespan', planner='cbs_', folder='outputs_pikachu', plot_median=False):
    """
    Generic plotting function for different metrics across environments
    
    Args:
        metric (str): Which metric to plot
        output_prefix (str): Prefix for output filenames
    """
    environments = ['dual_gp4', 'panda_two', 'panda_two_rod', 'panda_four', 'panda_four_bins', 'panda_three']
    dt_dict = {
        'dual_gp4': 0.025,
        'panda_two': 0.05,
        'panda_two_rod': 0.05,
        'panda_four': 1.0,
        'panda_four_bins': 1.0,
        'panda_three': 1.0
    }
    title_dict = {
        'dual_gp4': 'GP4 Two',
        'panda_two': 'Panda Two',
        'panda_two_rod': 'Panda Two Rod',
        'panda_four': 'Panda Four',
        'panda_four_bins': 'Panda Four Bins',
        'panda_three': 'Panda Three'
    }
    
    # Define methods to compare with distinct colors and markers
    # Main algorithms highlighted with bold colors, others dimmed but colorful
    entries = [
        ('#A4B0BE', 'X', f'{planner}comp_loose', 'Composite'),  # Dimmed blue-gray
        ('#D1A4B0', 'P', f'{planner}pp_loose', 'Prioritized'),  # Dimmed pink
        ('#95AFC0', 'o', f'{planner}path_loose', 'Path'),  # Dimmed blue
        ('#BDC581', 'd', f'{planner}random_loose', 'TPG'),  # Dimmed olive
        ('#A5B1C2', 'v', f'{planner}fwd_diter_loose', 'Fwd Loop'),  # Dimmed slate
        ('#B0C4DE', '>', f'{planner}bwd_diter_loose', 'Bwd Loop'),  # Dimmed silver
        ('#4444FF', '^', f'{planner}rr_loose', 'RR'),  # Bold blue
        #('#FF4444', 's', f'{planner}auto_loose', 'Weighted'),  # Bold red
        ('#009B77', 'D', f'{planner}thompson_loose', 'Thompson'),  # Bold emerald green
    ]

    # Configure metric-specific settings
    metric_configs = {
        'makespan': {
            'y_label': 'Makespan Improvement (%)',
            'data_key': 'makespan_improv_median' if plot_median else 'makespan_improv_mean',
            'y_lim': (0, None),
            'title': 'Makespan Improvement vs Time',
        },
        'flowtime': {
            'y_label': 'Flowtime Improvement (%)',
            'data_key': 'flowtime_median' if plot_median else 'flowtime_post_mean',
            'y_lim': (0, None),
            'title': 'Flowtime Improvement vs Time',
        },
        'pathlen': {
            'y_label': 'Sum of Path Length Improvement (%)',
            'data_key': 'pathlen_improv_median' if plot_median else 'pathlen_mean',
            'y_lim': (None, None),
            'title': 'Path Length Improvement vs Time',
        },
        'directional': {
            'y_label': 'Directional Consistency Improvement (%)',
            'data_key': 'dir_consistency_improv_median' if plot_median else 'dir_consistency_mean',
            'y_lim': (None, None),
            'title': 'Directional Consistency vs Time',
        },
        'n_valid': {
            'y_label': 'Number of Valid Shortcuts',
            'data_key': 'n_valid_median' if plot_median else 'n_valid_mean',
            'y_lim': (0, None),
            'title': 'Number of Valid Shortcuts vs Time',
        },
        'n_check': {
            'y_label': 'Number of Shortcut Checks',
            'data_key': 'n_check_median' if plot_median else 'n_check_mean',
            'y_lim': (0, None),
            'title': 'Number of Shortcut Checks vs Time',
        },
    }
    
    config = metric_configs[metric]

    # Create figure with larger size - adjusted for 3x2 layout
    plt.figure(figsize=(6, 7))
    
    # Update font sizes with correct parameter names
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 20
    })

    base_dir = os.path.dirname(os.path.realpath(__file__))

    # Create legend with increased size
    legend_fig = plt.figure(figsize=(20, 1))
    legend_ax = legend_fig.add_subplot(111)
    legend_lines = []
    legend_labels = []
    
    for color, marker, _, label in entries:
        is_highlighted = color in ['#FF4444', '#4444FF', '#009B77']
        line, = legend_ax.plot([], [], color=color, marker=marker, 
                             label=label, markersize=15, 
                             linewidth=3 if is_highlighted else 2)
        legend_lines.append(line)
        legend_labels.append(label)

    # Create legend with larger font size and more spacing
    legend = legend_ax.legend(legend_lines, legend_labels, 
                            loc='center', 
                            ncol=9,  # Reduced number of columns for better spacing
                            bbox_to_anchor=(0.5, 0.5),
                            fontsize=24,  # Significantly increased font size
                            handletextpad=0.5,  # Space between marker and text
                            columnspacing=2.0,  # Space between columns
                            handlelength=2.0)  # Length of the line in the legend
    # Make the legend lines longer
    for line in legend.get_lines():
        line.set_linewidth(3.0) 

    legend_ax.axis('off')
    legend_fig.savefig(os.path.join(base_dir, f'../outputs/plots/legend_{metric}.pdf'), 
                      bbox_inches='tight', dpi=300)
    plt.close(legend_fig)

    # Main plotting with larger markers and line widths
    for idx, env in enumerate(environments, 1):
        plt.subplot(3, 2, idx)
        plt.title(f'{title_dict[env]}', pad=10)
        
        for color, marker, algo, label in entries:
            dir = os.path.join(base_dir, f'../{folder}/{algo}')
            try:
                df = read_csv(dir, env)
                grouped_stats = df.groupby('timestep').apply(compute_statistics).reset_index()
                t_val_actual = grouped_stats['timestep'] * dt_dict[env]
                
                values = grouped_stats[config['data_key']][:-1]
                
                # Increase line width and marker size for highlighted algorithms
                is_highlighted = color in ['#FF4444', '#4444FF', '#009B77']
                linewidth = 2 if is_highlighted else 2
                markersize = 6 if is_highlighted else 6
                alpha = 1.0 if is_highlighted else 0.9
                
                plt.plot(t_val_actual[:-1], values, 
                        color=color, marker=marker,
                        markersize=markersize, markevery=0.2,
                        linewidth=linewidth, alpha=alpha)
                
            except Exception as e:
                print(f"Error plotting {algo} for {env}: {e}")
                continue

        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        if idx in [5, 6]:  # Adjusted for 3x2 layout
            plt.xlabel('Time (s)')
        if idx in [3]:  # Adjusted for 3x2 layout
            plt.ylabel(config['y_label'])
        if config['y_lim']:
            plt.ylim(*config['y_lim'])

    plt.tight_layout()
    
    plt.savefig(os.path.join(base_dir, f'../outputs/plots/{metric}_{planner}_comparison.pdf'), 
                bbox_inches='tight', dpi=500)
    plt.show()

def calculate_makespan_difference(folder='outputs'):
    """
    Calculates the average percentage difference in makespan_pre between two CSV files,
    grouping by start_pose and goal_pose.

    Args:
        csv_data1: CSV data as a string.
        csv_data2: CSV data as a string.

    Returns:
        A dictionary where keys are (start_pose, goal_pose) tuples and values are the
        average percentage difference in makespan_pre.  Returns an empty dictionary
        if errors occur or no matching pairs are found.
    """
    import io
    base_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(base_dir, f'../{folder}')

    t1, t2 = 0.0, 6.0
    planner = 'RRTstar'
    env = 'panda_two'
    
    file_path1 = os.path.join(folder, f't={t2}_{planner}/{env}_benchmark.csv')
    #file_path2 = os.path.join(folder, f't={t2}_{planner}/{env}_benchmark.csv')
    file_path2 = os.path.join(folder, f'thompson_loose/{env}_benchmark.csv')

    with open(file_path1, 'r') as f:
        csv_data1 = f.read()
    with open(file_path2, 'r') as f:
        csv_data2 = f.read()

    try:
        df1 = pd.read_csv(io.StringIO(csv_data1))
        df2 = pd.read_csv(io.StringIO(csv_data2))

        # Convert to numeric, setting invalid parses to NaN
        df1['makespan_post'] = pd.to_numeric(df1['makespan_post'], errors='coerce')
        df2['makespan_post'] = pd.to_numeric(df2['makespan_post'], errors='coerce')

        # Merge the dataframes based on start_pose and goal_pose
        merged_df = pd.merge(df1, df2, on=['start_pose', 'goal_pose'], suffixes=('_df1', '_df2'))

        # Drop rows where makespan_post is NaN in either dataframe
        merged_df = merged_df.dropna(subset=['makespan_post_df1', 'makespan_post_df2'])

        # Calculate percentage difference
        merged_df['makespan_diff'] = (
            (merged_df['makespan_post_df1'] -merged_df['makespan_post_df2']) / merged_df['makespan_post_df1']
        ) * 100

        # Group by start_pose and goal_pose and calculate the average difference
        grouped_diff = merged_df.groupby(['start_pose', 'goal_pose'])['makespan_diff'].mean()
        print(grouped_diff)
        # print number of unique (start_pose, goal_pose) pairs in merged dataframes
        print(f"Number of unique (start_pose, goal_pose) pairs: {len(df1.groupby(['start_pose', 'goal_pose']).size())}")
        print(grouped_diff.mean())

        return grouped_diff

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def generate_latex_tables(folder='outputs_pikachu'):
    """Generate LaTeX tables for different metrics across all methods and environments, alternating TPG and CBS rows"""
    environments = ['GP4 Two', 'Panda Two', 'Panda Two Rod', 'Panda Four', 'Panda Four Bins', 'Panda Three']
    env_file_names = {
        'GP4 Two': 'dual_gp4',
        'Panda Two': 'panda_two',
        'Panda Two Rod': 'panda_two_rod',
        'Panda Four': 'panda_four',
        'Panda Four Bins': 'panda_four_bins',
        'Panda Three': 'panda_three'
    }
    
    base_methods = [
        ('comp_loose', 'Composite'),
        ('pp_loose', 'Prioritized'),
        ('path_loose', 'Path'),
        ('random_loose', 'TPG'),
        ('fwd_diter_loose', 'Fwd Loop'),
        ('bwd_diter_loose', 'Bwd Loop'),
        #('auto_loose', 'Adaptive'),
        ('rr_loose', 'RR'),
        ('thompson_loose', 'DTS'),
    ]

    metrics = {
        'makespan': {
            'title': 'Makespan Improvement (\%)',
            'mean_key': 'makespan_improv_mean',
            'std_key': 'makespan_improv_std',
            'best': 'max',
            'decimal_places': 1,
        },
        'path_length': {
            'title': 'Path Length Improvement (\%)',
            'mean_key': 'pathlen_improv_mean',
            'std_key': 'pathlen_improv_std',
            'best': 'max',
            'decimal_places': 1
        },
        'dir_consistency': {
            'title': 'Directional Consistency Improvement (\%)',
            'mean_key': 'dir_consistency_improv_mean',
            'std_key': 'dir_consistency_improv_std',
            'best': 'max',
            'decimal_places': 1
        },
        'n_colcheck': {
            'title': 'Number of Collision Checks',
            'mean_key': 'n_colcheck_post_mean',
            'std_key': 'n_colcheck_post_std',
            'decimal_places': 0,
            'format_func': lambda x: f'{x / 1000:.0f}k'
        },
        'n_check': {
            'title': 'Number of Shortcuts Checked',
            'mean_key': 'n_check_mean',
            'std_key': 'n_check_std',
            'decimal_places': 0
        },
        'n_valid': {
            'title': 'Number of Valid Shortcuts',
            'mean_key': 'n_valid_mean',
            'std_key': 'n_valid_std',
            'best': 'max',
            'decimal_places': 0
        },
        'makespan_per_step': {
            'title': 'Average Makespan Improvement per Step',
            'mean_key': 'makespan_diff_per_step_mean',
            'std_key': 'makespan_diff_per_step_std',
            'best': 'max',
            'decimal_places': 2
        }
    }

    # Initialize data structure
    data = {metric: {} for metric in metrics}
    for metric in metrics:
        data[metric] = {'tpg': {}, 'cbs': {}}
        for method_type in ['tpg', 'cbs']:
            for method_id, _ in base_methods:
                data[metric][method_type][method_id] = {'mean': {}, 'std': {}}
    
    base_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Data collection (same as before)
    print("Collecting data...")
    for env_display, env_file in env_file_names.items():
        print(f"Processing environment: {env_display}")
        for method_id, method_name in base_methods:
            # Process TPG version
            try:
                dir_path = os.path.join(base_dir, f'../{folder}/{method_id}')
                df = read_csv(dir_path, env_file)
                grouped_stats = df.groupby('timestep').apply(compute_statistics).reset_index()
                
                for metric_name, metric_config in metrics.items():
                    mean_value = grouped_stats[metric_config['mean_key']].iloc[-2]
                    std_value = grouped_stats[metric_config['std_key']].iloc[-2]
                    
                    data[metric_name]['tpg'][method_id]['mean'][env_display] = mean_value
                    data[metric_name]['tpg'][method_id]['std'][env_display] = std_value
            except Exception as e:
                print(f"Error processing TPG {method_id} for {env_display}: {e}")
                for metric_name in metrics:
                    data[metric_name]['tpg'][method_id]['mean'][env_display] = float('nan')
                    data[metric_name]['tpg'][method_id]['std'][env_display] = float('nan')

            # Process CBS version
            try:
                dir_path = os.path.join(base_dir, f'../{folder}/cbs_{method_id}')
                df = read_csv(dir_path, env_file)
                grouped_stats = df.groupby('timestep').apply(compute_statistics).reset_index()
                
                for metric_name, metric_config in metrics.items():
                    mean_value = grouped_stats[metric_config['mean_key']].iloc[-2]
                    std_value = grouped_stats[metric_config['std_key']].iloc[-2]
                    
                    data[metric_name]['cbs'][method_id]['mean'][env_display] = mean_value
                    data[metric_name]['cbs'][method_id]['std'][env_display] = std_value
            except Exception as e:
                print(f"Error processing CBS {method_id} for {env_display}: {e}")
                for metric_name in metrics:
                    data[metric_name]['cbs'][method_id]['mean'][env_display] = float('nan')
                    data[metric_name]['cbs'][method_id]['std'][env_display] = float('nan')


    for metric_name, metric_config in metrics.items():
        output_file = os.path.join(base_dir, f'../outputs/{metric_name}_tables.tex')
        print(f"Generating LaTeX tables in {output_file}")
        with open(output_file, 'w') as f:
            # Find best values separately for each planner type and environment
            best_values = {
                'cbs': {env: float('-inf') if metric_config.get('best') == 'max' else float('inf') 
                        for env in environments},
                'tpg': {env: float('-inf') if metric_config.get('best') == 'max' else float('inf') 
                        for env in environments}
            }
            
            # Find the best values for each planner type separately
            if 'best' in metric_config:
                for planner in ['cbs', 'tpg']:
                    for env in environments:
                        for method_id, _ in base_methods:
                            value = data[metric_name][planner][method_id]['mean'][env]
                            if not np.isnan(value):
                                if metric_config['best'] == 'max':
                                    best_values[planner][env] = max(best_values[planner][env], value)
                                else:
                                    best_values[planner][env] = min(best_values[planner][env], value)

            f.write(f"% {metric_config['title']}\n")
            f.write("\\begin{tabular}{l|l|" + "c" * len(environments) + "}\n")
            f.write("\\toprule\n")
            
            f.write("Planner & Method & " + " & ".join(environments) + " \\\\\n")
            f.write("\\midrule\n")
            
            # First all CBS methods
            for i, (method_id, method_name) in enumerate(base_methods):
                row_values = []
                for env in environments:
                    mean = data[metric_name]['cbs'][method_id]['mean'][env]
                    std = data[metric_name]['cbs'][method_id]['std'][env]
                    if np.isnan(mean):
                        formatted_value = "-"
                    else:
                        if 'format_func' in metric_config:
                            mean_str = metric_config['format_func'](mean)
                            std_str = metric_config['format_func'](std)
                            formatted_value = f"{mean_str} $\\pm$ {std_str}"
                        else:
                            format_str = f"{{:.{metric_config['decimal_places']}f}}"
                            mean_str = format_str.format(mean)
                            std_str = format_str.format(std)
                            if 'best' in metric_config and abs(mean - best_values['cbs'][env]) < 1e-6:
                                mean_str = f"\\textbf{{{mean_str}}}"
                            formatted_value = f"{mean_str} $\\pm$ {std_str}"
                    row_values.append(formatted_value)
                
                if i == 0:
                    f.write(f"CBS & {method_name} & {' & '.join(row_values)} \\\\\n")
                else:
                    f.write(f"& {method_name} & {' & '.join(row_values)} \\\\\n")
                
                if method_id in ['path_loose', 'bwd_diter_loose']:
                    f.write("\\midrule\n")
            
            f.write("\\midrule\n")
            
            # Then all RRT methods
            for i, (method_id, method_name) in enumerate(base_methods):
                row_values = []
                for env in environments:
                    mean = data[metric_name]['tpg'][method_id]['mean'][env]
                    std = data[metric_name]['tpg'][method_id]['std'][env]
                    if np.isnan(mean):
                        formatted_value = "-"
                    else:
                        if 'format_func' in metric_config:
                            mean_str = metric_config['format_func'](mean)
                            std_str = metric_config['format_func'](std)
                            formatted_value = f"{mean_str} $\\pm$ {std_str}"
                        else:
                            format_str = f"{{:.{metric_config['decimal_places']}f}}"
                            mean_str = format_str.format(mean)
                            std_str = format_str.format(std)
                            if 'best' in metric_config and abs(mean - best_values['tpg'][env]) < 1e-6:
                                mean_str = f"\\textbf{{{mean_str}}}"
                            formatted_value = f"{mean_str} $\\pm$ {std_str}"
                    row_values.append(formatted_value)
                
                if i == 0:
                    f.write(f"RRT & {method_name} & {' & '.join(row_values)} \\\\\n")
                else:
                    f.write(f"& {method_name} & {' & '.join(row_values)} \\\\\n")
                
                if method_id in ['path_loose', 'bwd_diter_loose']:
                    f.write("\\midrule\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")

        print(f"LaTeX tables have been written to {output_file}")

def generate_aggregate_statistics(folder='outputs_pikachu'):
    """
    Generate LaTeX table with statistics aggregated across all environments.
    Using makecell for multiline headers, without collision check metrics.
    """
    environments = ['dual_gp4', 'panda_two', 'panda_two_rod', 'panda_four', 'panda_four_bins', 'panda_three']
    
    base_dir = os.path.dirname(os.path.realpath(__file__))
    
    base_methods = [
        ('comp_loose', 'Composite'),
        ('pp_loose', 'Prioritized'),
        ('path_loose', 'Path'),
        ('random_loose', 'TPG'),
        ('fwd_diter_loose', 'Fwd Loop'),
        ('bwd_diter_loose', 'Bwd Loop'),
        ('rr_loose', 'RR'),
        ('thompson_loose', 'DTS'),
    ]

    # Format function for k-notation
    def format_in_k(x):
        return f'{x/1000:.1f}k'
    
    def format_in_k2(x):
        return f'{x/1000:.0f}k'

    # Initialize data structure to store aggregated results
    aggregated_data = {}
    for planner in ['cbs', 'tpg']:
        aggregated_data[planner] = {}
        for method_id, _ in base_methods:
            aggregated_data[planner][method_id] = {
                'makespan_improv': {'values': []},
                'pathlen_improv': {'values': []},
                'dir_consistency_improv': {'values': []},
                'n_check': {'values': []},
                'n_valid': {'values': []},
                'makespan_diff_per_step': {'values': []},
            }

    # Collect data across all environments
    print("Collecting data...")
    for env in environments:
        print(f"Processing environment: {env}")
        for method_id, _ in base_methods:
            # Process both planner types
            for planner_type in ['cbs', 'tpg']:
                prefix = 'cbs_' if planner_type == 'cbs' else ''
                try:
                    dir_path = os.path.join(base_dir, f'../{folder}/{prefix}{method_id}')
                    df = read_csv(dir_path, env)
                    
                    # Get final values for each start_pose, goal_pose pair
                    final_values = df.groupby(['start_pose', 'goal_pose']).nth(-2)
                    
                    # Store values for each metric
                    metrics = ['makespan_improv', 'pathlen_improv', 'dir_consistency_improv', 
                             'n_check', 'n_valid', 'makespan_diff_per_step']
                    for metric in metrics:
                        if metric in final_values.columns:
                            aggregated_data[planner_type][method_id][metric]['values'].extend(
                                final_values[metric].values
                            )
                            print(f"Processed {planner_type} {method_id} {metric} for {env}")
                            
                except Exception as e:
                    print(f"Error processing {planner_type} {method_id} for {env}: {e}")
                    continue

    # Generate LaTeX table
    output_file = os.path.join(base_dir, f'../outputs/aggregate_statistics.tex')
    print(f"Generating LaTeX table in {output_file}")
    
    with open(output_file, 'w') as f:
        # Define column format with bold vertical line after third metric
        f.write("\\begin{tabular}{ll|ccc!{\\vrule width 1pt}ccc}\n")
        f.write("\\toprule\n")
        
        # Write header with makecell
        headers = [
            "Planner",
            "Method",
            "\\makecell{Makespan\\\\Improvement\\\\(\\%)}",
            "\\makecell{Path Length\\\\Improvement\\\\(\\%)}",
            "\\makecell{Directional\\\\Consistency\\\\Improvement(\\%)}",
            "\\makecell{\\# Candidate\\\\Shortcuts}",
            "\\makecell{\\# Valid\\\\Shortcuts}",
            "\\makecell{Makespan\\\\Improvement per\\\\ Valid Shortcuts}",
        ]
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        
        # Write data for each planner type and method
        for planner_display, planner_type in [("CBS", "cbs"), ("RRT", "tpg")]:
            # Find best values for each metric within this planner type
            metrics = ['makespan_improv', 'pathlen_improv', 'dir_consistency_improv', 
                      'n_check', 'n_valid', 'makespan_diff_per_step']
            best_values = {metric: None for metric in metrics}
            for method_id, _ in base_methods:
                for metric in metrics:
                    values = aggregated_data[planner_type][method_id][metric]['values']
                    if values:
                        mean_val = np.mean(values)
                        if best_values[metric] is None:
                            best_values[metric] = mean_val
                        elif metric in ['n_check']:  # minimize these metrics
                            best_values[metric] = min(best_values[metric], mean_val)
                        else:  # maximize other metrics
                            best_values[metric] = max(best_values[metric], mean_val)

            first_method = True
            for method_id, method_name in base_methods:
                # Process each metric
                row_values = []
                metrics_data = [
                    ('makespan_improv', 1, None),
                    ('pathlen_improv', 1, None),
                    ('dir_consistency_improv', 1, None),
                    ('n_check', None, format_in_k),
                    ('n_valid', 0, lambda x: f'{int(x)}'),
                    ('makespan_diff_per_step', 3, None),
                ]
                
                for metric, decimals, format_func in metrics_data:
                    values = aggregated_data[planner_type][method_id][metric]['values']
                    if values:
                        # exclude NaN values
                        values = [val for val in values if not np.isnan(val)]
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        
                        if format_func:
                            mean_str = format_func(mean_val)
                            std_str = format_func(std_val)
                        else:
                            mean_str = f"{mean_val:.{decimals}f}"
                            std_str = f"{std_val:.{decimals}f}"
                        
                        # Bold if this is the best value
                        is_best = abs(mean_val - best_values[metric]) < 1e-6
                        if is_best and metric not in ['n_check', 'makespan_diff_per_step', 'n_valid']:
                            formatted_value = f"\\textbf{{{mean_str} $\\pm$ {std_str}}}"
                        else:
                            formatted_value = f"{mean_str} $\\pm$ {std_str}"
                    else:
                        formatted_value = "-"
                    
                    row_values.append(formatted_value)

                # Write the row
                if first_method:
                    f.write(f"{planner_display} & {method_name} & {' & '.join(row_values)} \\\\\n")
                    first_method = False
                else:
                    f.write(f"& {method_name} & {' & '.join(row_values)} \\\\\n")
                
                # Add midrule between groups
                if method_id in ['path_loose', 'bwd_diter_loose']:
                    f.write("\\midrule\n")
            
            # Add midrule between planners
            if planner_type == "cbs":
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


from matplotlib.ticker import FuncFormatter

def plot_method_selection(planner='cbs_thompson', folder='outputs_pikachu', plot_valid=False, acculumate=True, prefix=""):
    """Plot the distribution of method selection for Auto method over time"""
    environments = ['dual_gp4', 'panda_two', 'panda_two_rod', 'panda_four', 'panda_four_bins', 'panda_three']
    dt_dict = {
        'dual_gp4': 0.025,
        'panda_two': 0.05,
        'panda_two_rod': 0.05,
        'panda_four': 1.0,
        'panda_four_bins': 1.0,
        'panda_three': 1.0
    }
    title_dict = {
        'dual_gp4': 'GP4 Two',
        'panda_two': 'Panda Two',
        'panda_two_rod': 'Panda Two Rod',
        'panda_four': 'Panda Four',
        'panda_four_bins': 'Panda Four Bins',
        'panda_three': 'Panda Three'
    }
    
    # Use consistent colors with the comparison plot
    colors = {
        'n_pp': '#FF4444',     # Bold red for Prioritized
        'n_path': '#4444FF',   # Bold blue for Path
        'n_comp': '#009B77',   # Bold green for Composite
        'n_v_pp': '#FF4444',   # Same colors for valid shortcuts
        'n_v_path': '#4444FF',
        'n_v_comp': '#009B77'
    }
    
    components = ['n_v_pp', 'n_v_path', 'n_v_comp'] if plot_valid else ['n_pp', 'n_path', 'n_comp']

    component_names = {
        'n_pp': 'Prioritized', 
        'n_path': 'Path', 
        'n_comp': 'Composite',
        'n_v_pp': 'Prioritized', 
        'n_v_path': 'Path', 
        'n_v_comp': 'Composite'
    }

    base_dir = os.path.dirname(os.path.realpath(__file__))

    # Create legend with increased size
    legend_fig = plt.figure(figsize=(10, 1))  # Increased height
    legend_ax = legend_fig.add_subplot(111)
    
    # Create patches for legend with larger sizes
    patches = [plt.Rectangle((0,0), 1, 1, fc=colors[comp]) for comp in components]
    labels = [component_names[comp] for comp in components]
    
    # Create legend with larger font size
    legend = legend_ax.legend(patches, labels, 
                            loc='center', 
                            ncol=3,  # Reduced columns for better spacing
                            bbox_to_anchor=(0.5, 0.5),
                            fontsize=24,  # Increased font size
                            handletextpad=0.5,
                            columnspacing=2.0)
    legend_ax.axis('off')
    
    # Save legend with higher DPI
    legend_fig.savefig(os.path.join(base_dir, f'../outputs/plots/method_selection_legend.pdf'), 
                      bbox_inches='tight', dpi=500)
    plt.close(legend_fig)

    # Update font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 20
    })

    # Create main figure
    fig, axs = plt.subplots(2, 3, figsize=(9, 4))
    axs = axs.flatten()

    # Formatter for y-axis
    def y_formatter(x, pos):
        if plot_valid:
            return f'{x:.0f}'
        else:
            if x >= 100:
                return f'{x/1000:.1f}k'
            else:
                return f'{x:.1f}'

    # For each environment
    for env_idx, env in enumerate(environments):
        ax = axs[env_idx]
        
        try:
            # Read and process data
            dir_path = os.path.join(base_dir, f'../{folder}/{planner}_loose')
            df = read_csv(dir_path, env)
            grouped_stats = df.groupby('timestep').apply(compute_statistics).reset_index()
            t_val_actual = grouped_stats['timestep'] * dt_dict[env]
            
            # Get the component percentages
            if acculumate:
                total = grouped_stats['n_valid_mean'].values[:-1] if plot_valid else grouped_stats['n_check_mean'].values[:-1]
                y_values = [grouped_stats[f'{comp}_mean'].values[:-1] * total for comp in components]
            else:
                y_values = [grouped_stats[f'{comp}_mean'].values[:-1] for comp in components]
            
            # Create stacked area plot
            ax.stackplot(t_val_actual[:-1], y_values, 
                        colors=[colors[comp] for comp in components],
                        alpha=0.7)  # Added some transparency
            
        except Exception as e:
            print(f"Error processing {planner} for {env}: {e}")
            continue
        
        # Customize subplot
        ax.set_title(title_dict[env], pad=5)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.2)
        if acculumate:
            ax.set_ylim(0, None)
        else:
            ax.set_ylim(0, 1)
        
        # Set y-axis formatter
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        
        # Only show x-axis labels for bottom plots
        if env_idx in [3, 4, 5]:
            ax.set_xlabel('Time (s)')
        
        # Adjust tick parameters for better visibility
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Add overarching y-axis label
    fig.text(0.013, 0.5, '# Shortcuts Sampled' if not plot_valid else '# Valid Shortcuts', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout()
    
    # Save the main figure
    plt.savefig(os.path.join(base_dir, f'../outputs/plots/{planner}{"_valid" if plot_valid else ""}_method_selection.pdf'), 
                bbox_inches='tight', dpi=500)
    plt.show()

def plot_method_scatter_comparison(baseline_method='path_loose', compare_method='auto_loose', metric='makespan', folder="outputs_pikachu"):
    """
    Create scatter plots comparing performance between two methods.
    Each point represents one problem instance.
    
    Args:
        baseline_method: Method to use as baseline (x-axis)
        compare_method: Method to compare against (y-axis)
        metric: Which metric to compare ('makespan' for makespan improvement)
    """
    environments = ['dual_gp4', 'panda_two', 'panda_two_rod', 'panda_four', 'panda_four_bins', 'panda_three']
    planners = ['cbs_', '']
    
    # Create figure with square aspect ratio
    plt.figure(figsize=(8, 8))
    
    # Define colors for different environments
    env_colors = {
        'dual_gp4': '#E69F00',
        'panda_two': '#56B4E9',
        'panda_two_rod': '#009E73',
        'panda_four': '#F0E442',
        'panda_four_bins': '#CC79A7',
        'panda_three': '#0072B2'
    }

    legend_map = {
        'path_loose': 'Path',
        'auto_loose': 'Weighted Discrete',
        'comp_loose': 'Composite',
        'random_loose': 'TPG',
        'pp_loose': 'Prioritized',
        'rr_loose': 'Round Robin',
        'thompson_loose': 'Thompson',
        'fwd_diter_loose': 'Forward Double',
        'bwd_diter_loose': 'Backward Double',
        'iter_loose': 'Iterative'
    }
    
    base_dir = os.path.dirname(os.path.realpath(__file__))
    all_x = []
    all_y = []
    
    # For each environment
    for env in environments:
        for planner in planners:
            try:
                # Read baseline method data
                baseline_dir = os.path.join(base_dir, f'../{folder}/{planner}{baseline_method}')
                baseline_df = read_csv(baseline_dir, env)
                
                # Read comparison method data
                compare_dir = os.path.join(base_dir, f'../{folder}/{planner}{compare_method}')
                compare_df = read_csv(compare_dir, env)
                print(env, planner, len(baseline_df), len(compare_df))

                # Get final makespan improvement for each start_pose, goal_pose pair
                baseline_final = baseline_df.groupby(['start_pose', 'goal_pose']).nth(-2)
                compare_final = compare_df.groupby(['start_pose', 'goal_pose']).nth(-2)
                
                # Extract makespan improvements
                x_vals = baseline_final['makespan_improv'].values
                y_vals = compare_final['makespan_improv'].values
                
                # Store for computing overall min/max
                all_x.extend(x_vals)
                all_y.extend(y_vals)
                
                # Plot points for this environment
                plt.scatter(x_vals, y_vals, 
                        #color=env_colors[env],
                        color='#56B4E9',
                        alpha=0.6,
                        #label=env.replace('_', ' ')
                )
                
            except Exception as e:
                print(f"Error processing {env}: {e}")
                continue
    
    # Plot diagonal line
    min_val = min(min(all_x), min(all_y))
    max_val = max(max(all_x), max(all_y))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Fit a linear regression line
    if all_x and all_y:
        coeffs = np.polyfit(all_x, all_y, 1)
        poly_eq = np.poly1d(coeffs)
        plt.plot([min_val, max_val], poly_eq([min_val, max_val]), 'r-')
        
        # Add annotation for the fitted line equation
        annotation_text = f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}'
        plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=12, color='red', ha='left', va='top')

    # Set equal aspect ratio
    plt.axis('square')
    
    # Labels and title
    plt.xlabel(f'Improvement (%) with {legend_map[baseline_method]}')
    plt.ylabel(f'Improvement (%) with {legend_map[compare_method]}')
    
    # Add grid
    plt.grid(True, alpha=0.2)
    
    # Add legend
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(base_dir, f'../outputs/plots/scatter_{baseline_method}_{compare_method}.png'), 
                bbox_inches='tight', dpi=300)
    plt.show()

def plot_multi_method_scatter(folder="outputs_pikachu", metric="makespan_improv", transpose=False):
    """
    Create a grid of scatter plots comparing multiple methods.
    Args:
        folder: Directory containing the data
        metric: Metric to plot (e.g., "makespan_improv")
        transpose: If True, flips the grid from 2x3 to 3x2 and switches x/y axes
    """
    # Define the comparison methods (rows by default)
    compare_methods = ['thompson_loose']
    # Define the baseline methods (columns by default)
    baseline_methods = ['path_loose', 'pp_loose', 'comp_loose']
    
    # Method display names for labels
    legend_map = {
        'path_loose': 'Path',
        'auto_loose': 'Adaptive',
        'comp_loose': 'Composite',
        'random_loose': 'TPG',
        'pp_loose': 'Prioritized',
        'rr_loose': 'Round Robin',
        'thompson_loose': 'DTS',
        'fwd_diter_loose': 'Fwd Loop',
        'bwd_diter_loose': 'Bwd Loop',
        'iter_loose': 'Iterative'
    }
    
    environments = ['dual_gp4', 'panda_two', 'panda_two_rod', 'panda_four', 'panda_four_bins', 'panda_three']
    planners = ['cbs_', '']
    
    # Determine plot dimensions based on transpose parameter
    if transpose:
        rows, cols = len(baseline_methods), len(compare_methods)
        figsize = (3 * len(compare_methods), 3 * len(baseline_methods))
    else:
        rows, cols = len(compare_methods), len(baseline_methods)
        figsize = (3 * len(baseline_methods), 3 * len(compare_methods))
    
    # Create figure with subplots
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Iterate over methods
    for i in range(rows):
        for j in range(cols):
            # Determine which methods to compare based on transpose setting
            if transpose:
                baseline_method = baseline_methods[i]
                compare_method = compare_methods[j]
            else:
                compare_method = compare_methods[i]
                baseline_method = baseline_methods[j]
            
            ax = axs[i, j] if (rows > 1 and cols > 1) else (axs[i] if rows > 1 else axs[j])
            all_x = []
            all_y = []
            
            # For each environment
            for env in environments:
                for planner in planners:
                    try:
                        # Read baseline method data
                        baseline_dir = os.path.join(base_dir, f'../{folder}/{planner}{baseline_method}')
                        baseline_df = read_csv(baseline_dir, env)
                        
                        # Read comparison method data
                        compare_dir = os.path.join(base_dir, f'../{folder}/{planner}{compare_method}')
                        compare_df = read_csv(compare_dir, env)
                        
                        # Get final makespan improvement for each start_pose, goal_pose pair
                        baseline_final = baseline_df.groupby(['start_pose', 'goal_pose']).nth(-2)
                        compare_final = compare_df.groupby(['start_pose', 'goal_pose']).nth(-2)
                        
                        # Extract makespan improvements
                        if transpose:
                            x_vals = compare_final[metric].values
                            y_vals = baseline_final[metric].values
                        else:
                            x_vals = baseline_final[metric].values
                            y_vals = compare_final[metric].values

                        # Store for computing overall min/max
                        all_x.extend(x_vals)
                        all_y.extend(y_vals)
                        
                        # Plot points for this environment
                        ax.scatter(x_vals, y_vals, 
                                color='#56B4E9',
                                alpha=0.6,
                                s=6)
                        
                    except Exception as e:
                        print(f"Error processing {env} for {baseline_method} vs {compare_method}: {e}")
                        continue
            
            if all_x and all_y:
                # Get the maximum value for both axes
                max_val = max(max(all_x), max(all_y))
                
                # Set axis limits from 0 to max_val
                ax.set_xlim(0, max_val)
                ax.set_ylim(0, max_val)
                
                # Plot diagonal line
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                
                # Fit a linear regression line
                coeffs = np.polyfit(all_x, all_y, 1)
                poly_eq = np.poly1d(coeffs)
                ax.plot([0, max_val], poly_eq([0, max_val]), 'r-')
                
                # Add annotation for the fitted line equation
                annotation_text = f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}'
                ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                          fontsize=14, color='red', ha='left', va='top')
            
                # Set equal aspect ratio
                ax.set_aspect('equal')
            
            # Add grid
            ax.grid(True, alpha=0.2)
            
            # Set labels based on transpose setting
            if transpose:
                if i == rows - 1:  # Only bottom row gets x-labels
                    ax.set_xlabel(f'{legend_map[compare_method]}', fontsize=12)
                if j == 0:  # Only leftmost column gets y-labels
                    ax.set_ylabel(f'{legend_map[baseline_method]}', fontsize=12)
            else:
                if i == rows - 1:  # Only bottom row gets x-labels
                    ax.set_xlabel(f'{legend_map[baseline_method]}', fontsize=16)
                if j == 0:  # Only leftmost column gets y-labels
                    ax.set_ylabel(f'{legend_map[compare_method]}', fontsize=16)

             # Set tick size
            ax.tick_params(axis='both', which='major', labelsize=14)  # Set major tick size
            ax.tick_params(axis='both', which='minor', labelsize=10)  # Set minor tick size
    
    plt.tight_layout()
    
    # Add overarching x-axis and y-axis labels with appropriate text based on transpose
    if transpose:
        fig.text(0.5, 0.02, 'Makespan Improvement (%) with Multi-Strategy Method', ha='center', va='center', fontsize=14)
        fig.text(0.02, 0.5, 'Makespan Improvement (%) with Randomized Methods', ha='center', va='center', rotation='vertical', fontsize=14)
        plt.subplots_adjust(left=0.14, bottom=0.1)
    else:
        fig.text(0.5, 0.032, 'Makespan Improvement (%)', ha='center', va='center', fontsize=17)
        fig.text(0.027, 0.5, 'Makespan \n Improvement (%)', ha='center', va='center', rotation='vertical', fontsize=17)
    
        # Adjust layout to make space for the new labels
        plt.subplots_adjust(left=0.12, bottom=0.32)
    
    # Save the figure
    orientation = "transposed" if transpose else "original"
    plt.savefig(os.path.join(base_dir, f'../outputs/plots/multi_method_scatter_{metric}_{orientation}.pdf'), 
                bbox_inches='tight')
    plt.show()

def compute_initial_statistics(folder='outputs_pikachu'):
    """
    Compute initial statistics (path length, makespan, directional consistency)
    for both CBS and RRT planners across all environments using the existing read_csv function.
    """
    # Define environments and metrics
    environments = ['dual_gp4', 'panda_two', 'panda_two_rod', 
                   'panda_four', 'panda_four_bins', 'panda_three']
    env_display = {
        'dual_gp4': 'GP4 Two',
        'panda_two': 'Panda Two', 
        'panda_two_rod': 'Panda Two Rod',
        'panda_four': 'Panda Four',
        'panda_four_bins': 'Panda Four Bins',
        'panda_three': 'Panda Three'
    }
    metrics = {
        'pathlen_pre': 'Path Length',
        'makespan_pre': 'Makespan',
        'dir_consistency_pre': 'Directional Consistency'
    }
    
    # Initialize results dictionary
    results = []
    
    # Get base directory
    base_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Process each environment
    for env in environments:
        # Process both CBS and RRT/TPG versions
        for planner_type in ['CBS', 'RRT']:
            # Use path_loose as reference implementation for each planner
            if planner_type == 'CBS':
                dir_path = os.path.join(base_dir, f'../{folder}/cbs_path_loose')
            else:
                dir_path = os.path.join(base_dir, f'../{folder}/path_loose')
                
            try:
                # Use the existing read_csv function to process the data
                df = read_csv(dir_path, env)
                
                # Get the first value for each start_pose, goal_pose pair
                initial_values = df.groupby(['start_pose', 'goal_pose']).first()
                
                # Compute statistics for each metric
                for metric, metric_name in metrics.items():
                    if metric in initial_values.columns:
                        mean_val = initial_values[metric].mean()
                        std_val = initial_values[metric].std()
                        
                        results.append({
                            'Environment': env_display[env],
                            'Planner': planner_type,
                            'Metric': metric_name,
                            'Mean': mean_val,
                            'Std': std_val,
                            'Count': len(initial_values)  # Add count of unique problems
                        })
                    
            except Exception as e:
                print(f"Error processing {env} for {planner_type}: {e}")
                
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Write results to file
    output_path = os.path.join(base_dir, '../outputs/initial_statistics.txt')
    with open(output_path, 'w') as f:
        # Write formatted results
        f.write("Initial Statistics Analysis\n")
        f.write("==========================\n\n")
        
        # Group by environment
        for env in env_display.values():
            f.write(f"{env}\n")
            f.write("-" * len(env) + "\n")
            
            env_data = results_df[results_df['Environment'] == env]
            for metric in metrics.values():
                f.write(f"\n{metric}:\n")
                metric_data = env_data[env_data['Metric'] == metric]
                
                for planner in ['CBS', 'RRT']:
                    planner_data = metric_data[metric_data['Planner'] == planner]
                    if not planner_data.empty:
                        mean_val = planner_data['Mean'].iloc[0]
                        std_val = planner_data['Std'].iloc[0]
                        count = planner_data['Count'].iloc[0]
                        f.write(f"{planner} ({count} problems): {mean_val:.3f} ± {std_val:.3f}\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
    print(f"Results written to {output_path}")


if __name__ == '__main__':
    import fire
    
    fire.Fire({
        'table': generate_latex_tables,
        'agg_table': generate_aggregate_statistics,
        'plot': plot,
        'plot_comp': plot_comparison,
        'selection': plot_method_selection,
        'scatter': plot_method_scatter_comparison,
        'multi_scatter': plot_multi_method_scatter,
        'initial': compute_initial_statistics,
        'calc_diff': calculate_makespan_difference
    })