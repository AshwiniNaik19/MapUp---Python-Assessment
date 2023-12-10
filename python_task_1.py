import numpy as np
import pandas as pd

#Question 1: Car Matrix Generation
def generate_car_matrix(df):
    # Pivot the dataframe
    result_df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    for idx in result_df.index:
        result_df.loc[idx, idx] = 0

    return result_df 
    
df=pd.read_csv("/home/mglocadmin/Downloads/Interview_taskk/python/MapUp-Data-Assessment-F-main/datasets/dataset-1.csv")
result_matrix = generate_car_matrix(df)
print(result_matrix)

#Question 2: Car Type Count Calculation
def get_type_count(df):
    # Create a new column 'car_type' based on the specified conditions
    df['car_type'] = np.select([df['car'] <= 15, (df['car'] > 15) & (df['car'] <= 25), df['car'] > 25],
                              ['low', 'medium', 'high'], default=None)
    
    # Display the DataFrame
    # print(df)

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))

    return type_count

df=pd.read_csv("/home/mglocadmin/Downloads/Interview_taskk/python/MapUp-Data-Assessment-F-main/datasets/dataset-1.csv")
result_type_count = get_type_count(df)
print(result_type_count)

#Question 3: Bus Count Index Retrieval
def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    mean_bus = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

df=pd.read_csv("/home/mglocadmin/Downloads/Interview_taskk/python/MapUp-Data-Assessment-F-main/datasets/dataset-1.csv")
result_bus_indexes = get_bus_indexes(df)
print(result_bus_indexes)

#Question 4: Route Filtering
def filter_routes(df):
    # Calculate the average 'truck' values for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # # Filter routes where the average 'truck' values are greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].tolist()

    # # Sort the list of routes
    filtered_routes.sort()

    return filtered_routes

df=pd.read_csv("/home/mglocadmin/Downloads/Interview_taskk/python/MapUp-Data-Assessment-F-main/datasets/dataset-1.csv")
result_filtered_routes = filter_routes(df)
print(result_filtered_routes)

#Question 5: Matrix Value Modification
def multiply_matrix(matrix):
    # Apply conditions and modify values
    matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    
    # Round values to 1 decimal place
    matrix = matrix.round(1)

    return matrix

modified_df = multiply_matrix(result_matrix)
print(modified_df)

#Question 6: Time Check
def time_check(df: pd.DataFrame) -> pd.Series:
    # Combine 'startDay' and 'startTime' columns to create a 'startTimestamp' column
    df['startTimestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')

    # Combine 'endDay' and 'endTime' columns to create an 'endTimestamp' column
    df['endTimestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Drop rows with missing timestamps
    df.dropna(subset=['startTimestamp', 'endTimestamp'], inplace=True)

    # Calculate the duration for each record
    df['duration'] = df['endTimestamp'] - df['startTimestamp']

    # Create a mask for incorrect timestamps
    mask = (df['startTimestamp'].dt.time != pd.Timestamp('00:00:00').time()) | \
           (df['endTimestamp'].dt.time != pd.Timestamp('23:59:59').time()) | \
           (df['duration'] != pd.Timedelta(days=6, hours=23, minutes=59, seconds=59))

    # Group by ('id', 'id_2') and check if any record has incorrect timestamps
    result_series = df.groupby(['id', 'id_2']).apply(lambda group: any(mask[group.index]))

    return result_series

df= pd.read_csv("/home/mglocadmin/Downloads/Interview_taskk/python/MapUp-Data-Assessment-F-main/datasets/dataset-2.csv")
result = time_check(df)
print(result)