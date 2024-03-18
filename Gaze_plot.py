import pandas as pd
from math import sqrt

# Load the dataset
df = pd.read_csv('GazeRV_ON26.csv')


# Function to calculate the distance between two points
def calculate_distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_mean_distance_per_time(total_distances_percentage, total_time):
    # Convert total times to seconds if they are in milliseconds
    total_times_in_seconds = total_time / 1000.0  # Adjust this line based on your data's time units
    mean_distance_per_time = total_distances_percentage / total_times_in_seconds
    return mean_distance_per_time

# Prepare a list to collect summary data for all participants
summary_data = []

for participant_id in df['IDParticipant'].unique():
    participant_data = df[df['IDParticipant'] == participant_id].sort_values(by='Time')
    
    total_distance = 0
    total_time = 0
    path_count = 0
    last_valid_point = None

    for i in range(len(participant_data) - 1):
        current_point = participant_data.iloc[i]
        next_point = participant_data.iloc[i + 1]
        
        if 0 <= current_point['ScreenX'] <= 1 and 0 <= current_point['ScreenY'] <= 1:
            if last_valid_point is not None:
                total_distance += calculate_distance(current_point['ScreenX'], current_point['ScreenY'],
                                                     last_valid_point['ScreenX'], last_valid_point['ScreenY'])
                total_time += next_point['Time'] - current_point['Time']
            last_valid_point = current_point
        else:
            path_count += 1
            last_valid_point = None

    if last_valid_point is not None and 0 <= participant_data.iloc[-1]['ScreenX'] <= 1 and 0 <= participant_data.iloc[-1]['ScreenY'] <= 1:
        total_time += participant_data.iloc[-1]['Time'] - last_valid_point['Time']


    # Assuming total_distance_percentage and total_time_percentage are calculated as before
    mean_distance_per_time_percentage = calculate_mean_distance_per_time(total_distance, total_time)

    # Append the collected information for the current participant to the summary list
    summary_data.append({
        'Participant ID': participant_id,
        'Total Distance': total_distance,
        'Total Time in ms': total_time,
        'mean_distance_per_sec' : mean_distance_per_time_percentage,
        'Path Count': path_count + 1  # Adjust for the last path
    })



# Convert the summary list to a DataFrame
summary_df = pd.DataFrame(summary_data)

# Set the Participant ID as the index (optional)
summary_df.set_index('Participant ID', inplace=True)

# Print the DataFrame as a table
print(summary_df)
