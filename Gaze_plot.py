import pandas as pd
from math import sqrt
pd.set_option('display.max_columns', 100)
df = pd.read_csv('GazeRV.csv', low_memory=False)

def calculate_distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_mean_distance_per_time(total_distance, total_time):
    total_times_in_seconds = total_time / 1000.0  
    if total_times_in_seconds == 0:  # Prévention de la division par zéro
        return 0
    mean_distance_per_time = total_distance / total_times_in_seconds
    return mean_distance_per_time

summary_data = []

for participant_id in df['IDParticipant'].unique():
    participant_data = df[df['IDParticipant'] == participant_id].sort_values(by='Time')
    
    total_distance = 0
    total_distance_outside = 0
    total_time = 0
    total_time_outside = 0  # Temps total passé en dehors de l'écran
    path_count = 0
    last_valid_point = None
    last_outside_point = None
    out_of_bounds = False

    for i in range(len(participant_data) - 1):
        current_point = participant_data.iloc[i]
        next_point = participant_data.iloc[i + 1]
        
        if 0 <= current_point['ScreenX'] <= 1 and 0 <= current_point['ScreenY'] <= 1:
            if last_valid_point is not None:
                total_distance += calculate_distance(current_point['ScreenX'], current_point['ScreenY'],
                                                     last_valid_point['ScreenX'], last_valid_point['ScreenY'])
                total_time += next_point['Time'] - current_point['Time']
            last_valid_point = current_point
            if out_of_bounds:
                path_count += 1
                out_of_bounds = False
        else:
            if last_outside_point is not None:
                total_distance_outside += calculate_distance(current_point['ScreenX'], current_point['ScreenY'],
                                                             last_outside_point['ScreenX'], last_outside_point['ScreenY'])
                total_time_outside += next_point['Time'] - current_point['Time']
            last_outside_point = current_point
            if not out_of_bounds:
                out_of_bounds = True
            last_valid_point = None

    mean_distance_per_time_inside = calculate_mean_distance_per_time(total_distance, total_time)
    mean_distance_per_time_outside = calculate_mean_distance_per_time(total_distance_outside, total_time_outside)

    summary_data.append({
        'Participant ID': participant_id,
        'Total Distance': total_distance,
        'Total Distance Outside': total_distance_outside,
        'Total Time in ms': total_time,
        'Total Time Outside in ms': total_time_outside,  # Ajout du temps passé en dehors de l'écran
        'Mean Distance per Second Inside': mean_distance_per_time_inside,
        'Mean Distance per Second Outside': mean_distance_per_time_outside,  # Moyenne de la distance par temps en dehors
        'Path Count': path_count
    })


summary_df = pd.DataFrame(summary_data)

summary_df.set_index('Participant ID', inplace=True)

print(summary_df)


total_mean_inside = sum([d['Mean Distance per Second Inside'] for d in summary_data]) / len(summary_data)
total_mean_outside = sum([d['Mean Distance per Second Outside'] for d in summary_data]) / len(summary_data)
global_mean = (total_mean_inside + total_mean_outside) / 2
total_path_count = sum([d['Path Count'] for d in summary_data])


print(f'Moyenne globale de la distance Inside: {total_mean_inside}')
print(f'Moyenne globale de la distance Outside: {total_mean_outside}')
print(f'Moyenne globale de la distance par seconde: {global_mean}')
print(f'Total Path Count pour tous les participants: {total_path_count}')
