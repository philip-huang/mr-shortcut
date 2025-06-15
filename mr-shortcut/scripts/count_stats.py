import pandas as pd

# Read the CSV data into a DataFrame

# Define a function to calculate the average improvement
def average_improvement(df, out_df=None):
    # Add percentage improvement columns to the DataFrame
    df['flowtime_improvement_perc'] = ((df['flowtime_pre'] - df['flowtime_post']) / df['flowtime_pre']) * 100
    df['makespan_improvement_perc'] = ((df['makespan_pre'] - df['makespan_post']) / df['makespan_pre']) * 100

    # Calculate the average percentage improvements
    avg_flowtime_improvement_perc = df['flowtime_improvement_perc'].mean()
    avg_makespan_improvement_perc = df['makespan_improvement_perc'].mean()

    # Append a row with the averages
    # calculate mean and average
    mean = df.select_dtypes(include=['float64', 'int64']).mean()
    std = df.select_dtypes(include=['float64', 'int64']).std()
    df.loc['Average'] = mean
    df.loc['Std Dev'] = std

    if out_df:
        df.to_csv(out_df, sep=',')
    
    return df

# Calculate the average improvements

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input csv file")
    ap.add_argument("-o", "--output", required=False, help="path to output csv file")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=',')

    print(average_improvement(df, out_df=args.output))