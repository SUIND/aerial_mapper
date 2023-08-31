
import pyexiv2
import os

import utm
import numpy as np

import math
import random
import pandas as pd


def to_quaternion(roll, pitch, yaw):
    # Abbreviations for the various angular functions
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) - np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
 
    return [qw, qx, qy, qz]

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    quart= np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    quart= quart/(np.sqrt(quart[0]**2+quart[1]**2+quart[2]**2+quart[3]**2))
    return quart


folder_path = "/home/axelwagner/catkin_ws_aerial_mapper/src/camera_extract"

drone_pos_txt = []

north, east = 0,0
height_0 = 0
csv_file_path = "/home/axelwagner/DJI_RECORDINGS/match/Aug-31st-2023-05-34PM-Flight-Airdata.csv"
csv_data = pd.read_csv(csv_file_path)
# for column in csv_data.columns:
#     print(column)
# compass_heading(degrees), pitch(degrees), roll(degrees)
# gimbal_roll = row["gimbal_roll(degrees)"]
#     gimbal_pitch = row["gimbal_pitch(degrees)"]
#     gimbal_heading = row["gimbal_heading(degrees)"]
# 0.00670337315266709 0.961768673473662 0.239566286796365 -0.132529535363593
for i, row in csv_data.iterrows():
    latitude = row["latitude"]
    longitude = row["longitude"]
    gimbal_roll = row["gimbal_roll(degrees)"]
    gimbal_pitch = row["gimbal_pitch(degrees)"]
    gimbal_heading = row["gimbal_heading(degrees)"]
    altitude_above_seaLevel = row["height_above_takeoff(feet)"]
    # if (i < 50):
    #     print(altitude_above_seaLevel*0.3048)

    euler_Gimbal = [float(gimbal_roll) * (math.pi / 180),
                    float(gimbal_pitch) * (math.pi / 180),
                    float(gimbal_heading) * (math.pi / 180)]
    result_quaternion_gimbal = get_quaternion_from_euler(euler_Gimbal[0], euler_Gimbal[1], euler_Gimbal[2])
    rot = [0.707, 0.0, 0.707, 0.0]
    rot_2 = [0.0, 1.0, 0.0, 0.0]
    result_quaternion_gimbal = quaternion_multiply(rot,result_quaternion_gimbal)
    result_quaternion_gimbal = quaternion_multiply(rot_2,result_quaternion_gimbal)
    
    height = float(altitude_above_seaLevel) * 0.3048  # Convert feet to meters
    UTM_E, UTM_N, Zone, _ = utm.from_latlon(float(latitude), float(longitude))

    if i == 0:
        north, east = UTM_N, UTM_E
        # height_0 = height


    UTM_E = UTM_E - east
    UTM_N = UTM_N - north
    # height = height -height_0

    tmp_list = [UTM_E, UTM_N, height, result_quaternion_gimbal[0], result_quaternion_gimbal[1],
                result_quaternion_gimbal[2], result_quaternion_gimbal[3]]  # Corrected order of elements
    # tmp_list = [UTM_E, UTM_N, height, 0.00670337315266709 ,0.961768673473662 ,0.239566286796365 ,-0.132529535363593]  # Corrected order of elements

    drone_pos_txt.append(tmp_list)

file_path = f'{folder_path}/{"opt_poses.txt"}'
with open(file_path, 'w') as file:
    for value_set in drone_pos_txt:
        line = " ".join(str(value) for value in value_set)
        file.write(line + '\n')


print("File 'data.txt' created successfully!")

# import yaml

# data ={
#     "label": "sensorpod: calibration hitimo",
#     "cameras": [
#         {
#             "camera": {
#                 "label": "cam0",
#                 "distortion": {
#                     "parameters": {
#                         "cols": 1,
#                         "rows": 4,
#                         "data": [0, 0, 0, 0]
#                     },
#                     "type": "equidistant"
#                 },
#                 "image_height": 480,
#                 "image_width": 752,
#                 "intrinsics": {
#                     "cols": 1,
#                     "rows": 4,
#                     "data": [7389.595375722543, 7243.538461538462, 2640, 1973]
#                 },
#                 "type": "pinhole",
#                 "line-delay-nanoseconds": 0
#             },
#             "T_B_C": {
#                 "cols": 4,
#                 "rows": 4,
#                 "data": [1, 0, 0, 0,
#                          0, 1, 0, 0,
#                           0, 0, 1, 0,
#                           0, 0, 0, 1]
#             }
#         }
#     ]
# }

# yaml_file_path = f'{folder_path}/{"calibration.yaml"}'
# with open(yaml_file_path, "w") as yaml_file:
#     yaml.dump(data, yaml_file, default_flow_style=False)



# Load the CSV files
file2_path = csv_file_path
file1_path = 'new_subtitles.csv'

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Find the row with the specified name in both DataFrames

target_row1 = df1["H"]
target_row2 = df2["height_above_takeoff(feet)"]

# Extract the values from the rows
values_row1 = target_row1.values.flatten()
values_row2 = target_row2.values.flatten()*0.3048

# Function to find the first value above a certain threshold in a row
def find_first_above_threshold(row, threshold):
    for value in row:
        if float(value.strip("m"))  > threshold:
            return float(value.strip("m") )
    return None  # If no value is found above the threshold

# Function to find the value closest to a target value in a row
def find_closest_value(row, target):
    return min(row, key=lambda x: abs(x - target))

# Define the threshold and target value
threshold = 10.0
target_value = find_first_above_threshold(values_row1, threshold)

# Find the first value above the threshold
closest_value = find_closest_value(values_row2, target_value)
closest_row_index = values_row2.tolist().index(closest_value)
closest_row = df2.iloc[closest_row_index]

print("Row closest to the target value:")
print(closest_row["time(millisecond)"])



import cv2

# Video file path
video_path = "/home/axelwagner/DJI_RECORDINGS/match/DJI_0364.MP4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Set the starting point (in milliseconds)
start_time_ms = closest_row["time(millisecond)"]

# Set the interval (in milliseconds)
interval_ms = 100

# Set the frame position to the starting point
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Calculate the starting frame index
start_frame = int(frame_rate * (start_time_ms / 1000))

# Calculate the frame interval based on frame rate
frame_interval = int(frame_rate * (interval_ms / 1000)+0.5)
print(frame_interval)
# Set the frame position to the starting frame
print(start_frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frames_used = 0
max_frames = 1300
for _ in range(start_frame):
    ret, _ = cap.read()
    if not ret:
        print("Reached the end of the video.")
        break
while cap.isOpened() and frames_used < max_frames:
    ret, frame = cap.read()

    if not ret:
        break

    # Display the frame or save it to a file
    cv2.imshow('Frame', frame)

    # Wait for the interval
    if cv2.waitKey(interval_ms) & 0xFF == ord('q'):
        break

    # Skip frames to match the interval
    for _ in range(frame_interval - 1):
        cap.grab()

    frame_filename = f'frames/frame_{frames_used}.jpg'
    cv2.imwrite(frame_filename, frame)

    frames_used += 1

cap.release()
cv2.destroyAllWindows()

print(frames_used)
