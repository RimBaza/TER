import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

df = pd.read_csv('Bureau.csv')


def calculate_distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_mean_distance_per_time(total_distances_percentage, total_time):
    total_times_in_seconds = total_time / 1000.0  
    mean_distance_per_time = total_distances_percentage / total_times_in_seconds
    return mean_distance_per_time

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


    mean_distance_per_time_percentage = calculate_mean_distance_per_time(total_distance, total_time)

    summary_data.append({
        'Participant ID': participant_id,
        'Total Distance': total_distance,
        'Total Time in ms': total_time,
        'mean_distance_per_sec' : mean_distance_per_time_percentage,
        'Path Count': path_count + 1  # Adjust for the last path
    })


summary_df = pd.DataFrame(summary_data)

summary_df.set_index('Participant ID', inplace=True)

print(summary_df)

metrics = ['Total Distance', 'Total Time in ms', 'mean_distance_per_sec', 'Path Count']
for metric in metrics:
    summary_df.plot(kind='bar', x='Participant ID', y=metric, figsize=(10, 6), legend=True)
    plt.title(f'{metric} for Each Participant')
    plt.ylabel(metric)
    plt.xlabel('Participant ID')
    plt.tight_layout()
    plt.show()

