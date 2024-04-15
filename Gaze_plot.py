import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN


data = pd.read_csv('GazeRV_ON26.csv')
data['Time'] = data['Time'] / 1000
data_grouped = data.groupby('IDParticipant')

# Parameters
time_window = 200  # milliseconds
dispersion_threshold = 25  # pixels
screen_bounds = (0, 1) 

# Function to calculate dispersion
def calculate_dispersion(subset):
    x_range = subset['ScreenX'].max() - subset['ScreenX'].min()
    y_range = subset['ScreenY'].max() - subset['ScreenY'].min()
    return max(x_range, y_range)

# Identifying fixations
def identify_fixations(data, time_window, dispersion_threshold):
    fixations = []
    start = 0
    data = data.sort_values('Time')  # Ensure the data is sorted by time
    
    while start < len(data):
        end = start
        current_window = data.iloc[start:end+1]
       

        # Increase the window size until the end of the data or the time window exceeds the limit
        while end < len(data) - 1 and (data.iloc[end+1]['Time'] - data.iloc[start]['Time']) <= time_window:
            end += 1
            current_window = data.iloc[start:end+1]
            if calculate_dispersion(current_window) > dispersion_threshold:
                break
        
        # If the dispersion within the final window is under the threshold, classify as a fixation
        if calculate_dispersion(current_window) <= dispersion_threshold:
            fixation_center_x = current_window['ScreenX'].mean()
            fixation_center_y = current_window['ScreenY'].mean()
            fixation_start_time = data.iloc[start]['Time']
            fixation_end_time = data.iloc[end]['Time']
            fixations.append((fixation_center_x, fixation_center_y, fixation_start_time, fixation_end_time))
        
        # Move to the next possible start point
        start = end + 1

    return fixations

def calculate_saccade_distances(fixations):
    distances = []
    for i in range(1, len(fixations)):
        prev_fixation = fixations[i-1]
        current_fixation = fixations[i]
        distance = math.sqrt((current_fixation[0] - prev_fixation[0])**2 + (current_fixation[1] - prev_fixation[1])**2)
        distances.append(distance)
    return distances




def calculate_scanpaths(data):
    total_scanpath_length = 0
    scanpath_count = 1  # Start with one scanpath by default
    data = data.sort_values('Time')  # Ensure the data is sorted by time

    for i in range(1, len(data)):
        x1, y1 = data.iloc[i-1]['ScreenX'], data.iloc[i-1]['ScreenY']
        x2, y2 = data.iloc[i]['ScreenX'], data.iloc[i]['ScreenY']
        
        # Check if the point is within screen bounds
        if not (screen_bounds[0] <= x1 <= screen_bounds[1] and screen_bounds[0] <= y1 <= screen_bounds[1]):
            scanpath_count += 1  # Increment scanpath count if out of bounds
            continue  # Skip out-of-bounds points
        
        # Calculate distance if within bounds
        if screen_bounds[0] <= x2 <= screen_bounds[1] and screen_bounds[0] <= y2 <= screen_bounds[1]:
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_scanpath_length += distance

    return total_scanpath_length, scanpath_count



# Apply fixation identification per participant and prepare plot data
fixation_counts = {}  # To store fixation counts for plotting
average_durations = {}
saccade_distances = {}
scanpath_info = {}  # To store scanpath lengths and counts

for participant_id, group in data_grouped:
    fixations = identify_fixations(group, time_window, dispersion_threshold)
    fixation_counts[participant_id] = len(fixations)
    fixation_df = pd.DataFrame(fixations, columns=['Center X', 'Center Y', 'Start Time', 'End Time'])
    fixation_df['Duration'] = fixation_df['End Time'] - fixation_df['Start Time']
    average_durations[participant_id] = fixation_df['Duration'].mean()
    distances = calculate_saccade_distances(fixations)
    saccade_distances[participant_id] = distances
    scanpath_length, scanpath_count = calculate_scanpaths(group)
    scanpath_info[participant_id] = (scanpath_length, scanpath_count)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

participants = list(fixation_counts.keys())
counts = list(fixation_counts.values())
ax1.bar(participants, counts, color='blue', label='Number of Fixations')
ax1.set_ylabel('Number of Fixations', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')

average_duration_values = list(average_durations.values())
ax1_twin = ax1.twinx()
ax1_twin.plot(participants, average_duration_values, color='red', marker='o', label='Average Fixation Duration (ms)')
ax1_twin.set_ylabel('Average Fixation Duration (ms)', color='red')
ax1_twin.tick_params(axis='y', labelcolor='red')
ax1_twin.legend(loc='upper right')

ax1.set_title('Number of Fixations and Average Fixation Duration per Participant')

saccade_means = {participant: np.mean(distances) if distances else 0 for participant, distances in saccade_distances.items()}
mean_distances = list(saccade_means.values())
ax2.bar(participants, mean_distances, color='green', label='Average Saccade Distance')
ax2.set_xlabel('Participant ID')
ax2.set_ylabel('Average Saccade Distance (pixels)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

ax2.set_title('Average Saccade Distance per Participant')


# Third figure: Plot each saccade's distance
fig3, ax3 = plt.subplots(figsize=(10, 6))
for participant_id, distances in saccade_distances.items():
    ax3.scatter([participant_id] * len(distances), distances, label=f'Participant {participant_id}')
ax3.set_xlabel('Participant ID')
ax3.set_ylabel('Saccade Distance (pixels)')
plt.title('Saccade Distances for Each Participant')
plt.legend(title="Participants", bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position outside the plot



# Plot 3: Total Scanpath Length and Number of Scanpaths
fig4, ax4 = plt.subplots(figsize=(10, 6))
scanpath_lengths = [info[0] for info in scanpath_info.values()]
scanpath_counts = [info[1] for info in scanpath_info.values()]

ax4.bar(participants, scanpath_lengths, color='purple', label='Total Scanpath Length')
ax4.set_ylabel('Total Scanpath Length (pixels)', color='purple')
ax4.tick_params(axis='y', labelcolor='purple')
ax4.legend(loc='upper left')

ax4 = ax4.twinx()
ax4.plot(participants, scanpath_counts, color='orange', marker='o', label='Number of Scanpaths')
ax4.set_ylabel('Number of Scanpaths', color='orange')
ax4.tick_params(axis='y', labelcolor='orange')
ax4.legend(loc='upper right')

ax4.set_title('Scanpath Metrics per Participant')


plt.tight_layout()
plt.show()


# Supposez que 'fixations' est un DataFrame contenant les centres X et Y de toutes les fixations
# Si 'fixations' n'existe pas déjà, vous devrez créer ce DataFrame à partir de 'fixation_df' pour chaque participant.
# Voici comment vous pourriez le faire (en supposant que les données sont déjà préparées dans le code initial) :

# Rassembler toutes les fixations dans un seul DataFrame (dépend de la structure de votre programme)
all_fixations = pd.concat([group[['ScreenX', 'ScreenY']] for _, group in data_grouped])

# Application de DBSCAN
# 'eps' est la distance maximale entre deux échantillons pour qu'ils soient considérés comme dans le même voisinage
# 'min_samples' est le nombre de points minimum dans un voisinage pour former un cluster
dbscan = DBSCAN(eps=0.05, min_samples=20)  # Ces paramètres doivent être ajustés en fonction de vos données spécifiques
clusters = dbscan.fit_predict(all_fixations)

# Ajouter les étiquettes de cluster au DataFrame
all_fixations['Cluster'] = clusters

filtered_data = all_fixations[all_fixations['Cluster'] != -1]
fixation_statistics = filtered_data.groupby('Cluster').agg(
    Avg_ScreenX=('ScreenX', 'mean'),
    Avg_ScreenY=('ScreenY', 'mean'),
    Duration=('Time', lambda x: x.max() - x.min()),
    Count=('Cluster', 'size')
).reset_index()
print(fixation_statistics)
# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(all_fixations['ScreenX'], all_fixations['ScreenY'], c=all_fixations['Cluster'], cmap='viridis', marker='o')
plt.title('DBSCAN Clustering of Fixation Points')
plt.xlabel('Screen X')
plt.ylabel('Screen Y')
plt.colorbar(label='Cluster ID')
plt.show()
