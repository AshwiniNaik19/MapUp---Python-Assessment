import pandas as pd
import numpy as np

#Question 1: Distance Matrix Calculation
def calculate_distance_matrix(df):
    # Create a list of unique locations
    locations = list(set(df['id_start'].unique()) | set(df['id_end'].unique()))

    # Create an empty dataframe with unique locations as both index and columns
    distance_df = pd.DataFrame(index=locations, columns=locations)

    # Fill the dataframe with cumulative distances along known routes
    for _, row in df.iterrows():
        distance_df.at[row['id_start'], row['id_end']] = row['distance']

    # Convert missing values to 0
    distance_df = distance_df.fillna(0)

    # Make the matrix symmetric
    distance_df = distance_df + distance_df.T

    # Set diagonal values to 0
    np.fill_diagonal(distance_df.values, 0)

    return distance_df

df= pd.read_csv("/home/mglocadmin/Downloads/Interview_taskk/python/MapUp-Data-Assessment-F-main/datasets/dataset-3.csv")
result = calculate_distance_matrix(df)
print(result)

#Question 2: Unroll Distance Matrix
def unroll_distance_matrix(df):
    # Extract unique locations from the distance matrix
    locations = df.index

    # Initialize an empty list to store unrolled data
    unrolled_data = []

    # Iterate through all combinations of locations
    for start_loc in locations:
        for end_loc in locations:
            # Skip combinations where start and end locations are the same
            if start_loc != end_loc:
                # Append the combination and corresponding distance to the list
                unrolled_data.append({
                    'id_start': start_loc,
                    'id_end': end_loc,
                    'distance': df.at[start_loc, end_loc]
                })

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

unrolled_result = unroll_distance_matrix(result)
print(unrolled_result)


#Question 3: Finding IDs within Percentage Threshold

def find_ids_within_ten_percentage_threshold(df):
    # Filter the DataFrame based on the reference_id
    for reference_id in df['id_start']:
        reference_df = df[df['id_start'] == reference_id]

        # Calculate the average distance for the reference_id
        reference_avg_distance = reference_df['distance'].mean()

        # Calculate the threshold for 10%
        threshold = 0.1 * reference_avg_distance

        # Filter IDs within the threshold
        result_df = df[
            (df['id_start'] != reference_id) &  # Exclude the reference_id
            (df['distance'] >= (reference_avg_distance - threshold)) &
            (df['distance'] <= (reference_avg_distance + threshold))
        ]

        # Sort the result by id_start column
        result_df = result_df.sort_values(by='id_start')

        return result_df

result = find_ids_within_ten_percentage_threshold(unrolled_result)
print(result)



#Question 4: Calculate Toll Rate

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

result2 = calculate_toll_rate(result)
print(result2)


#Question 5: Calculate Time-Based Toll Rates
import pandas as pd
from datetime import datetime, time

def calculate_time_based_toll_rates(df):
    # Define time ranges and discount factors
    time_ranges_weekday = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]

    time_ranges_weekend = [
        (time(0, 0, 0), time(23, 59, 59), 0.7)
    ]

    # Create a dictionary to map day indices to day names
    day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Initialize lists to store new columns
    start_day_list, start_time_list, end_day_list, end_time_list = [], [], [], []

    # Iterate through rows in the DataFrame
    for _, row in df.iterrows():
        # Extract start and end datetime objects
        start_datetime = datetime.strptime(f"{row['startDay']} {row['startTime']}", "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(f"{row['endDay']} {row['endTime']}", "%Y-%m-%d %H:%M:%S")

        # Get day index and map it to day name
        day_index = start_datetime.weekday()
        start_day_name = day_mapping[day_index]
        end_day_name = day_mapping[(day_index + 1) % 7]  # Next day

        # Append values to lists
        start_day_list.append(start_day_name)
        end_day_list.append(end_day_name)

        # Iterate through time ranges to calculate toll rates
        toll_rate = 0.0
        for start_time, end_time, discount_factor in (time_ranges_weekday if day_index < 5 else time_ranges_weekend):
            if start_datetime.time() >= start_time and end_datetime.time() <= end_time:
                toll_rate = row['distance'] * discount_factor
                break

        # Append start and end times to lists
        start_time_list.append(start_datetime.time())
        end_time_list.append(end_datetime.time())

    # Add new columns to the DataFrame
    df['start_day'] = start_day_list
    df['start_time'] = start_time_list
    df['end_day'] = end_day_list
    df['end_time'] = end_time_list

    return df

result3 = calculate_time_based_toll_rates(result2)
print(result3)
