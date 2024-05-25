import os
import pandas as pd

# Define the search path (current directory or specific base directory)
search_path = '.'  # Replace with the path to your base directory if different

# Dictionary to hold lists of DataFrames for each unique filename
data_frames_dict = {}

# Walk through the directory hierarchy
for root, dirs, files in os.walk(search_path):
    for dir_name in dirs:
        if dir_name.startswith('Outputs'):
            csv_folder_path = os.path.join(root, dir_name, 'CSVS')
            if os.path.exists(csv_folder_path):
                # List all CSV files in the CSVS folder
                csv_files = [file for file in os.listdir(csv_folder_path) if file.endswith('.csv')]
                for csv_file in csv_files:
                    csv_path = os.path.join(csv_folder_path, csv_file)
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if csv_file not in data_frames_dict:
                            data_frames_dict[csv_file] = []
                        data_frames_dict[csv_file].append(df)

# Create output directory for averaged CSVs
output_dir = os.path.join(search_path, 'AverageOutputs')
os.makedirs(output_dir, exist_ok=True)

# print(data_frames_dict)
# Check if we found any files
if not data_frames_dict:
    print("No CSV files found.")
else:
    # Process each set of DataFrames corresponding to each unique filename
    for filename, data_frames in data_frames_dict.items():
        # Ensure all DataFrames have the same columns
        columns = data_frames[0].columns
        for df in data_frames:
            if not df.columns.equals(columns):
                raise ValueError(f"CSV files with filename '{filename}' have different columns.")
        
        # Concatenate DataFrames and compute the row-wise average
        combined_df = pd.concat(data_frames).groupby(level=0).mean()
        
        # Save the averaged DataFrame to a new CSV file in the output directory
        output_file = os.path.join(output_dir, filename)
        combined_df.to_csv(output_file, index=False)
        print(f"Saved averaged CSV to {output_file}")


import os
import pandas as pd

# Define the search path (current directory or specific base directory)
search_path = '.'  # Replace with the path to your base directory if different

# Dictionary to hold lists of DataFrames for each unique filename
data_frames_dict = {}

# Walk through the directory hierarchy
for root, dirs, files in os.walk(search_path):
    for dir_name in dirs:
        if dir_name.startswith('Outputs'):
            csv_folder_path = os.path.join(root, dir_name, 'CSVS')
            if os.path.exists(csv_folder_path):
                # List all CSV files in the CSVS folder
                csv_files = [file for file in os.listdir(csv_folder_path) if file.endswith('.csv')]
                for csv_file in csv_files:
                    csv_path = os.path.join(csv_folder_path, csv_file)
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if csv_file not in data_frames_dict:
                            data_frames_dict[csv_file] = []
                        data_frames_dict[csv_file].append(df)

# Create output directory for MAXd CSVs
output_dir = os.path.join(search_path, 'MAXOutputs')
os.makedirs(output_dir, exist_ok=True)

# print(data_frames_dict)
# Check if we found any files
if not data_frames_dict:
    print("No CSV files found.")
else:
    # Process each set of DataFrames corresponding to each unique filename
    for filename, data_frames in data_frames_dict.items():
        # Ensure all DataFrames have the same columns
        columns = data_frames[0].columns
        for df in data_frames:
            if not df.columns.equals(columns):
                raise ValueError(f"CSV files with filename '{filename}' have different columns.")
        
        # Concatenate DataFrames and compute the row-wise MAX
        combined_df = pd.concat(data_frames).groupby(level=0).max()
        
        # Save the MAXd DataFrame to a new CSV file in the output directory
        output_file = os.path.join(output_dir, filename)
        combined_df.to_csv(output_file, index=False)
        print(f"Saved MAXd CSV to {output_file}")


import os
import pandas as pd

# Define the search path (current directory or specific base directory)
search_path = '.'  # Replace with the path to your base directory if different

# Dictionary to hold lists of DataFrames for each unique filename
data_frames_dict = {}

# Walk through the directory hierarchy
for root, dirs, files in os.walk(search_path):
    for dir_name in dirs:
        if dir_name.startswith('Outputs'):
            csv_folder_path = os.path.join(root, dir_name, 'CSVS')
            if os.path.exists(csv_folder_path):
                # List all CSV files in the CSVS folder
                csv_files = [file for file in os.listdir(csv_folder_path) if file.endswith('.csv')]
                for csv_file in csv_files:
                    csv_path = os.path.join(csv_folder_path, csv_file)
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if csv_file not in data_frames_dict:
                            data_frames_dict[csv_file] = []
                        data_frames_dict[csv_file].append(df)

# Create output directory for MINd CSVs
output_dir = os.path.join(search_path, 'MINOutputs')
os.makedirs(output_dir, exist_ok=True)

# print(data_frames_dict)
# Check if we found any files
if not data_frames_dict:
    print("No CSV files found.")
else:
    # Process each set of DataFrames corresponding to each unique filename
    for filename, data_frames in data_frames_dict.items():
        # Ensure all DataFrames have the same columns
        columns = data_frames[0].columns
        for df in data_frames:
            if not df.columns.equals(columns):
                raise ValueError(f"CSV files with filename '{filename}' have different columns.")
        
        # Concatenate DataFrames and compute the row-wise MIN
        combined_df = pd.concat(data_frames).groupby(level=0).min()
        
        # Save the MINd DataFrame to a new CSV file in the output directory
        output_file = os.path.join(output_dir, filename)
        combined_df.to_csv(output_file, index=False)
        print(f"Saved MINd CSV to {output_file}")


import pandas as pd


import os
if not os.path.exists('min_max_avg'):
    os.makedirs('min_max_avg')
def combine_csv(file1_path, file2_path, file3_path, output_path):
    # Read CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    df3 = pd.read_csv(file3_path)
    
    # Merge dataframes on the common first column
    merged_df = df1.merge(df2, on='Step').merge(df3, on='Step')
    
    # Rename columns to be more descriptive
    merged_df.columns = ['step', 'min', 'avg', 'max']
    
    # Write merged dataframe to CSV
    merged_df.to_csv(output_path, index=False)

file1_path = 'MINOutputs/clogv.csv'
file2_path = 'AverageOutputs/clogv.csv'
file3_path = 'MAXOutputs/clogv.csv'
output_path = 'min_max_avg/min_max_avg_clogv.csv'

combine_csv(file1_path, file2_path, file3_path, output_path)



file1_path = 'MINOutputs/gep.csv'
file2_path = 'AverageOutputs/gep.csv'
file3_path = 'MAXOutputs/gep.csv'
output_path = 'min_max_avg/min_max_avg_gep.csv'

combine_csv(file1_path, file2_path, file3_path, output_path)



file1_path = 'MINOutputs/gcvp.csv'
file2_path = 'AverageOutputs/gcvp.csv'
file3_path = 'MAXOutputs/gcvp.csv'
output_path = 'min_max_avg/min_max_avg_gcvp.csv'

combine_csv(file1_path, file2_path, file3_path, output_path)


file1_path = 'MINOutputs/rwsp.csv'
file2_path = 'AverageOutputs/rwsp.csv'
file3_path = 'MAXOutputs/rwsp.csv'
output_path = 'min_max_avg/min_max_avg_rwsp.csv'

combine_csv(file1_path, file2_path, file3_path, output_path)



file1_path = 'MINOutputs/ghc.csv'
file2_path = 'AverageOutputs/ghc.csv'
file3_path = 'MAXOutputs/ghc.csv'
output_path = 'min_max_avg/min_max_avg_ghc.csv'

combine_csv(file1_path, file2_path, file3_path, output_path)

import shutil
if not os.path.exists('plot_data'):
    os.makedirs('plot_data')


source_path = "MINOutputs"
destination_path = 'plot_data'
shutil.move(source_path, destination_path)



source_path = "AverageOutputs"
destination_path = 'plot_data'
shutil.move(source_path, destination_path)



source_path = "MAXOutputs"
destination_path = 'plot_data'
shutil.move(source_path, destination_path)



source_path = "min_max_avg"
destination_path = 'plot_data'
shutil.move(source_path, destination_path)



# Doing for GD

import pandas as pd
import matplotlib.pyplot as plt


def plot_range(csv_file1, csv_file2, output_file):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    # Extract data for plotting from the first CSV
    steps1 = df1.iloc[:, 0]
    min_values1 = df1.iloc[:, 1]
    avg_values1 = df1.iloc[:, 2]
    max_values1 = df1.iloc[:, 3]

    # Extract data for plotting from the second CSV
    steps2 = df2.iloc[:, 0]
    min_values2 = df2.iloc[:, 1]
    avg_values2 = df2.iloc[:, 2]
    max_values2 = df2.iloc[:, 3]

    # Plot the data
    plt.figure(figsize=(10, 6))

    # Plot the average lines
    plt.plot(steps1, avg_values1, color='red', label='GHC avg')
    plt.plot(steps2, avg_values2, color='blue', label='RWSP avg')

    # Plot the range (min to max) with shaded areas
    plt.fill_between(steps1, min_values1, max_values1, color='red', alpha=0.1, label='GHC min-max')
    plt.fill_between(steps2, min_values2, max_values2, color='blue', alpha=0.1, label='RWSP min-max')

    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.title('Range Plot with Average Lines and Min-Max Shading for GD')
    plt.legend()

    # Save plot to a file
    plt.savefig(output_file, format='pdf', dpi=600)

    # Optionally, close the plot to free up memory
    plt.close()

# Replace with your actual file paths and desired output file path
csv_file1 = 'plot_data/min_max_avg/min_max_avg_ghc.csv'
csv_file2 = 'plot_data/min_max_avg/min_max_avg_rwsp.csv'
output_file = 'plot_data/min_max_avg/gd.pdf'

plot_range(csv_file1, csv_file2, output_file)


# Doing for PD
import pandas as pd
import matplotlib.pyplot as plt

def plot_range(csv_file1, csv_file2, csv_file3, output_file):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    df3 = pd.read_csv(csv_file3)

    # Extract data for plotting from the first CSV
    steps1 = df1.iloc[:, 0]
    min_values1 = df1.iloc[:, 1]
    avg_values1 = df1.iloc[:, 2]
    max_values1 = df1.iloc[:, 3]

    # Extract data for plotting from the second CSV
    steps2 = df2.iloc[:, 0]
    min_values2 = df2.iloc[:, 1]
    avg_values2 = df2.iloc[:, 2]
    max_values2 = df2.iloc[:, 3]

    # Extract data for plotting from the third CSV
    steps3 = df3.iloc[:, 0]
    min_values3 = df3.iloc[:, 1]
    avg_values3 = df3.iloc[:, 2]
    max_values3 = df3.iloc[:, 3]

    # Plot the data
    plt.figure(figsize=(12, 8))

    # Plot the average lines
    plt.plot(steps1, avg_values1, color='green', label='Average CLOGV')
    plt.plot(steps2, avg_values2, color='red', label='Average GCVP')
    plt.plot(steps3, avg_values3, color='blue', label='Average GEP')

    # Plot the range (min to max) with shaded areas
    plt.fill_between(steps1, min_values1, max_values1, color='green', alpha=0.1, label='Min-Max CLOGV')
    plt.fill_between(steps2, min_values2, max_values2, color='red', alpha=0.1, label='Min-Max GCVP')
    plt.fill_between(steps3, min_values3, max_values3, color='blue', alpha=0.1, label='Min-Max GEP')

    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.title('Range Plot with Average Lines and Min-Max Shading for PD')
    plt.legend()

    # Save plot to a file
    plt.savefig(output_file, format='pdf', dpi=600)

    # Optionally, close the plot to free up memory
    plt.close()

# File paths for the input CSVs
csv_file1 = 'plot_data/min_max_avg/min_max_avg_clogv.csv'
csv_file2 = 'plot_data/min_max_avg/min_max_avg_gcvp.csv'
csv_file3 = 'plot_data/min_max_avg/min_max_avg_gep.csv'
output_file = 'plot_data/min_max_avg/pd.pdf'

# Call the function with the file paths
plot_range(csv_file1, csv_file2, csv_file3, output_file)


# Doing for singles and their corresponding range plots.






# CLOGV
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'plot_data/min_max_avg/min_max_avg_clogv.csv'
df = pd.read_csv(file_path)

# Extract data for plotting
steps = df.iloc[:, 0]
min_values = df.iloc[:, 1]
avg_values = df.iloc[:, 2]
max_values = df.iloc[:, 3]

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the average line
plt.plot(steps, avg_values, color='blue', label='CLOGV')

# Plot the range (min to max) with a shaded area
plt.fill_between(steps, min_values, max_values, color='blue', alpha=0.1, label='CLOGV Range')

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Range Plot CLOGV')
plt.legend()

# Save plot to a file
output_file = 'plot_data/min_max_avg/min_max_avg_clogv.pdf'
plt.savefig(output_file, format='pdf',dpi=600)

# Optionally, close the plot to free up memory
plt.close()




# GCVP
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'plot_data/min_max_avg/min_max_avg_gcvp.csv'
df = pd.read_csv(file_path)

# Extract data for plotting
steps = df.iloc[:, 0]
min_values = df.iloc[:, 1]
avg_values = df.iloc[:, 2]
max_values = df.iloc[:, 3]

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the average line
plt.plot(steps, avg_values, color='blue', label='GCVP')

# Plot the range (min to max) with a shaded area
plt.fill_between(steps, min_values, max_values, color='blue', alpha=0.1, label='GCVP Range')

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Range Plot GCVP')
plt.legend()

# Save plot to a file
output_file = 'plot_data/min_max_avg/min_max_avg_gcvp.pdf'
plt.savefig(output_file, format='pdf',dpi=600)

# Optionally, close the plot to free up memory
plt.close()



# GEP
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'plot_data/min_max_avg/min_max_avg_gep.csv'
df = pd.read_csv(file_path)

# Extract data for plotting
steps = df.iloc[:, 0]
min_values = df.iloc[:, 1]
avg_values = df.iloc[:, 2]
max_values = df.iloc[:, 3]

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the average line
plt.plot(steps, avg_values, color='blue', label='GEP')

# Plot the range (min to max) with a shaded area
plt.fill_between(steps, min_values, max_values, color='blue', alpha=0.1, label='GEP Range')

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Range Plot GEP')
plt.legend()

# Save plot to a file
output_file = 'plot_data/min_max_avg/min_max_avg_gep.pdf'
plt.savefig(output_file, format='pdf',dpi=600)

# Optionally, close the plot to free up memory
plt.close()



# RWSP
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'plot_data/min_max_avg/min_max_avg_rwsp.csv'
df = pd.read_csv(file_path)

# Extract data for plotting
steps = df.iloc[:, 0]
min_values = df.iloc[:, 1]
avg_values = df.iloc[:, 2]
max_values = df.iloc[:, 3]

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the average line
plt.plot(steps, avg_values, color='blue', label='RWSP')

# Plot the range (min to max) with a shaded area
plt.fill_between(steps, min_values, max_values, color='blue', alpha=0.1, label='RWSP Range')

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Range Plot RWSP')
plt.legend()

# Save plot to a file
output_file = 'plot_data/min_max_avg/min_max_avg_rwsp.pdf'
plt.savefig(output_file, format='pdf',dpi=600)

# Optionally, close the plot to free up memory
plt.close()



# GHC
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'plot_data/min_max_avg/min_max_avg_ghc.csv'
df = pd.read_csv(file_path)

# Extract data for plotting
steps = df.iloc[:, 0]
min_values = df.iloc[:, 1]
avg_values = df.iloc[:, 2]
max_values = df.iloc[:, 3]

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the average line
plt.plot(steps, avg_values, color='blue', label='GHC')

# Plot the range (min to max) with a shaded area
plt.fill_between(steps, min_values, max_values, color='blue', alpha=0.1, label='GHC Range')

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Range Plot GHC')
plt.legend()

# Save plot to a file
output_file = 'plot_data/min_max_avg/min_max_avg_ghc.pdf'
plt.savefig(output_file, format='pdf',dpi=600)

# Optionally, close the plot to free up memory
plt.close()



# PLOT DATA FOR AVERAGES
import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the folder containing CSV files
folder_path = 'plot_data/AverageOutputs'
output_folder = 'plot_data/AverageOutputs'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Loop through each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(folder_path, file))
    
    # Assuming the first column is 'x' and the rest are 'y' values
    x = df.iloc[:, 0]  # First column as x
    y_columns = df.columns[1:]  # All other columns as y values

    # Plot each y column
    plt.figure(figsize=(10, 6))  # Optional: specify figure size
    for y_col in y_columns:
        plt.plot(x, df[y_col], label=f'{y_col}')
    
    # Add labels and legend
    plt.xlabel('Steps')
    plt.ylabel(f'{file}')
    plt.title(f'Plot of {file}')
    plt.legend()
    
    # Save the plot to the output folder
    output_file = os.path.join(output_folder, f'{file}.pdf')
    plt.savefig(output_file, format='pdf', dpi=600)
    plt.close()  # Close the figure to free memory
