import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

pd.set_option('display.max_columns', 100)
df = pd.read_csv('C:/Users/fares/Desktop/Ter/TER/GazeRV_ON26.csv', low_memory=False)
'''
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




#Zones de fixation

# Function to clean and prepare data
def prepare_data(df):
    # Convert comma decimal points to dots
    for col in ['ScreenHeadDistance', 'GazeDistance']:
        df[col] = df[col].str.replace(',', '.').astype(float)

    return df


# Function to apply DBSCAN and find clusters (fixation zones)
def find_fixation_zones(df):
    # Coordinates need to be scaled as time is in different units (milliseconds)
    # Scaling factor to adjust the time so it has comparable magnitude to X and Y coordinates
    time_scaling_factor = 1e-6

    # Prepare features for DBSCAN
    features = df[['ScreenX', 'ScreenY', 'Time']] * [1, 1, time_scaling_factor]

    # Applying DBSCAN
    # eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples is the number of samples in a neighborhood for a point to be considered as a core point.
    clustering = DBSCAN(eps=0.15, min_samples=20).fit(features)

    # Add cluster labels to the dataframe
    df['Cluster'] = clustering.labels_

    return df


# Clean and prepare the data
clean_data = prepare_data(df)

# Find fixation zones
fixation_data = find_fixation_zones(clean_data)

# Filter out noise points by keeping only those with a cluster label >= 0
clusters = fixation_data[fixation_data['Cluster'] >= 0]

# Group data by clusters to get statistics
fixation_statistics = clusters.groupby('Cluster').agg(
    Avg_ScreenX=('ScreenX', 'mean'),
    Avg_ScreenY=('ScreenY', 'mean'),
    Duration=('Time', lambda x: x.max() - x.min()),
    Count=('Cluster', 'size')
).reset_index()

fixation_statistics.head()
print(fixation_statistics)





def calculate_dispersion(points):
    """Calculate the maximum dispersion (distance) within a group of points."""
    if len(points) < 2:
        return 0
    x_coords = points['ScreenX']
    y_coords = points['ScreenY']
    max_dispersion = np.sqrt((x_coords.max() - x_coords.min())**2 + (y_coords.max() - y_coords.min())**2)
    return max_dispersion

def identify_fixations(data, dispersion_threshold, min_duration):
    """Identify fixations based on the I-DT algorithm."""
    fixations = []
    fixation_points = pd.DataFrame(columns=data.columns)
    start_time = data.iloc[0]['Time']

    for i in range(len(data)):
        point = data.iloc[i]
        fixation_points = fixation_points._append(point, ignore_index=True)
        if calculate_dispersion(fixation_points) > dispersion_threshold:
            # Check the duration of the points before the last added point
            if (fixation_points.iloc[-2]['Time'] - start_time) >= min_duration:
                # Save the fixation
                fixation_stats = {
                    'Start_Time': start_time,
                    'End_Time': fixation_points.iloc[-2]['Time'],
                    'Duration': fixation_points.iloc[-2]['Time'] - start_time,
                    'Centroid_X': fixation_points['ScreenX'][:-1].mean(),
                    'Centroid_Y': fixation_points['ScreenY'][:-1].mean()
                }
                fixations.append(fixation_stats)
            # Start new fixation candidate
            start_time = point['Time']
            fixation_points = pd.DataFrame([point])

    # Check last fixation
    if len(fixation_points) > 1 and (fixation_points.iloc[-1]['Time'] - start_time) >= min_duration:
        fixation_stats = {
            'Start_Time': start_time,
            'End_Time': fixation_points.iloc[-1]['Time'],
            'Duration': fixation_points.iloc[-1]['Time'] - start_time,
            'Centroid_X': fixation_points['ScreenX'].mean(),
            'Centroid_Y': fixation_points['ScreenY'].mean()
        }
        fixations.append(fixation_stats)

    return pd.DataFrame(fixations)

# Example usage
# Load your dataset (make sure to adjust the path and column names according to your data)
dispersion_threshold = 0.5  # Set this based on your data and what is considered a single fixation area
min_duration = 200  # minimum duration in milliseconds for a fixation

fixations = identify_fixations(df, dispersion_threshold, min_duration)
print(fixations.head())
'''

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1['ScreenX'] - p2['ScreenX'])**2 + (p1['ScreenY'] - p2['ScreenY'])**2)

def temporal_spatial_region_query(data, index, eps_space, eps_time):
    """Query for neighbors within spatial and temporal eps thresholds."""
    point = data.iloc[index]
    neighbors = []

    # Only check forward in time to maintain temporal consecutiveness
    next_index = index + 1
    while next_index < len(data) and abs(data.iloc[next_index]['Time'] - point['Time']) <= eps_time:
        spatial_distance = np.linalg.norm(data.iloc[next_index][['ScreenX', 'ScreenY']] - point[['ScreenX', 'ScreenY']])
        if spatial_distance <= eps_space:
            neighbors.append(next_index)
        next_index += 1

    return neighbors

def expand_cluster(data, neighbors, cluster_id, eps_space, eps_time, minPts):
    """Expand the cluster to include density reachable points."""
    queue = list(neighbors)
    while queue:
        current_idx = queue.pop(0)
        if data.at[current_idx, 'Cluster'] == -1:
            data.at[current_idx, 'Cluster'] = cluster_id
            new_neighbors = temporal_spatial_region_query(data, current_idx, eps_space, eps_time)
            if len(new_neighbors) >= minPts:
                queue.extend(new_neighbors)  # Union of neighbors

def dbscan(data, eps_space, eps_time, minPts):
    """DBSCAN algorithm adjusted for temporal and spatial data."""
    data['Cluster'] = -1  # Initialize all points as noise (-1)
    cluster_id = 0

    for index in range(len(data)):
        if data.at[index, 'Cluster'] != -1:
            continue  # Skip already visited points
        neighbors = temporal_spatial_region_query(data, index, eps_space, eps_time)
        if len(neighbors) < minPts:
            data.at[index, 'Cluster'] = -1  # Mark as noise
        else:
            expand_cluster(data, neighbors, cluster_id, eps_space, eps_time, minPts)
            cluster_id += 1

    return data



# Example usage
eps_space = 0.05 # Spatial epsilon, assuming normalized coordinates and/or timing threshold
eps_time = 1000
minPts = 40  # Minimum number of points to form a cluster

# Apply DBSCAN
clustered_data = dbscan(df, eps_space, eps_time, minPts)
print(clustered_data['Cluster'].value_counts())
filtered_data = clustered_data[clustered_data['Cluster'] != -1]
fixation_statistics = filtered_data.groupby('Cluster').agg(
    Avg_ScreenX=('ScreenX', 'mean'),
    Avg_ScreenY=('ScreenY', 'mean'),
    Duration=('Time', lambda x: x.max() - x.min()),
    Count=('Cluster', 'size')
).reset_index()
print(fixation_statistics)

# Quick diagnostic to see if 'Time' varies within clusters
def check_time_variation_per_cluster(data):
    return data.groupby('Cluster')['Time'].nunique()

# Run this diagnostic
time_variations = check_time_variation_per_cluster(filtered_data)
print(time_variations)

'''
# Calculate differences between consecutive timestamps to understand their distribution
time_differences = df['Time'].diff().abs()
print("Time Differences Stats:")
print(time_differences.describe())

# Plot histogram of time differences to visually inspect the spread
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(time_differences.dropna(), bins=50, alpha=0.75)
plt.title('Histogram of Time Differences')
plt.xlabel('Time Differences')
plt.ylabel('Frequency')
plt.show()
'''