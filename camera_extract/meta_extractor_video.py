
import pyexiv2
import os
import utm
import numpy as np
import math
import random
import pandas as pd
import cv2
import subprocess
import re
import csv



# Set the path to the video file
video_path = '/home/axelwagner/Downloads/test_1/DJI_0354.MP4'

#current folder path
folder_path = "/home/axelwagner/catkin_ws_aerial_mapper/src/aerial_mapper/camera_extract"

#path to video csv
csv_file_path = "/home/axelwagner/DJI_RECORDINGS/match/Aug-31st-2023-05-34PM-Flight-Airdata.csv"


#Function to get quarternion from roll pitch yaw
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

#Function to do quaternion multiplication  
def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    quart= np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    quart= quart/(np.sqrt(quart[0]**2+quart[1]**2+quart[2]**2+quart[3]**2))
    return quart

folder_name = "frames"

# Check if the folder already exists
if not os.path.exists(folder_name):
    # If it doesn't exist, create it
    os.mkdir(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")

drone_pos_txt = []

north, east = 0,0
height_0 = 0

csv_data = pd.read_csv(csv_file_path)
#go trough csv line by line and extract the required information
for i, row in csv_data.iterrows():
    latitude = row["latitude"]
    longitude = row["longitude"]
    gimbal_roll = row["gimbal_roll(degrees)"]
    gimbal_pitch = row["gimbal_pitch(degrees)"]
    gimbal_heading = row["gimbal_heading(degrees)"]
    altitude_above_seaLevel = row["height_above_takeoff(feet)"]
    # if (i < 50):
    #     print(altitude_above_seaLevel*0.3048)

    #from degree to radiant
    euler_Gimbal = [float(gimbal_roll) * (math.pi / 180),
                    float(gimbal_pitch) * (math.pi / 180),
                    float(gimbal_heading) * (math.pi / 180)]
    #unchanged coordinates from flight data to quarternions 
    result_quaternion_gimbal = get_quaternion_from_euler(euler_Gimbal[0], euler_Gimbal[1], euler_Gimbal[2])

    #!!! these are quarternions we tried using to rotate the reference system
    rot = [0.707, 0.0, 0.707, 0.0]
    rot_2 = [0.0, 1.0, 0.0, 0.0]
    result_quaternion_gimbal = quaternion_multiply(rot,result_quaternion_gimbal)
    result_quaternion_gimbal = quaternion_multiply(rot_2,result_quaternion_gimbal)
    #!!!
    
    height = float(altitude_above_seaLevel) * 0.3048  # Convert feet to meters
    UTM_E, UTM_N, Zone, _ = utm.from_latlon(float(latitude), float(longitude)) # convert gps to UTM

    #set first position to 0, 0
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

#format op_poses
file_path = f'{folder_path}/{"frames/opt_poses.txt"}'
with open(file_path, 'w') as file:
    for value_set in drone_pos_txt:
        line = " ".join(str(value) for value in value_set)
        file.write(line + '\n')


print("File 'data.txt' created successfully!")


#Get subtitle csv
# Run FFmpeg command to extract subtitles
ffmpeg_command = [
    'ffmpeg',
    '-i', video_path,
    '-f', 'srt',  # Specify the output format as SubRip (srt)
    '-vn', '-an',  # Disable video and audio streams
    '-map', '0:s:0',  # Select the first subtitle stream
    '-y',  # Overwrite the output file if it already exists
    'output.srt'  # Output subtitle file
]

subprocess.run(ffmpeg_command)

# Read the srt file and split it into subtitle entries
with open('output.srt', 'r', encoding='utf-8') as srt_file:
    srt_content = srt_file.read()

# Split the srt content into individual subtitle entries
subtitle_entries = re.split(r'\n\n', srt_content)

# Initialize a CSV writer
csv_file = open('new_subtitles.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)

# Write CSV header
csv_writer.writerow(['Subtitle Number', 'Start Time', 'End Time', 'Subtitle Text'])

# Process each subtitle entry
for i, entry in enumerate(subtitle_entries):
    lines = entry.strip().split('\n')
    if len(lines) >= 3 and re.match(r'^\d+$', lines[0]):
        start_time, end_time = re.findall(r'(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
        subtitle_text = '\n'.join(lines[2:])
        csv_writer.writerow([i + 1, start_time, end_time, subtitle_text])

# Close the CSV file
csv_file.close()

# Read the CSV file into a DataFrame
df = pd.read_csv('new_subtitles.csv')

# Extract specific values from the "Subtitle Text" column
def extract_value(text, key):
    start_idx = text.find(key)
    if start_idx != -1:
        start_idx += len(key) + 1  # Move past the key and space
        if key == 'GPS':
            start_idx += 1  # Move past the opening parenthesis
            end_idx = text.find(')', start_idx)
            if end_idx == -1:
                end_idx = None
        elif key == 'D':
            if start_idx != -1:
                second_key_idx = text.find(key, start_idx + 1)
                if second_key_idx != -1:
                    start_idx = second_key_idx
                    start_idx += len(key) + 1  # Move past the key and space
                    end_idx = text.find(',', start_idx)
                    if end_idx == -1:
                        end_idx = None
        else:
            end_idx = text.find(',', start_idx)
            if end_idx == -1:
                end_idx = None
        return text[start_idx:end_idx].strip()
    return None

keys_to_extract = ['F', 'SS', 'ISO', 'EV', 'DZOOM', 'GPS', 'D', 'H', 'H.S', 'V.S']

for key in keys_to_extract:
    df[key] = df['Subtitle Text'].apply(lambda x: extract_value(x, key))

# Drop the "Subtitle Text" column
df.drop(columns=['Subtitle Text'], inplace=True)

# Save the modified DataFrame back to the final CSV file
df.to_csv('final_subtitles.csv', index=False)

# Print the first value for each column
for column in df.columns:
    first_value = df[column].iloc[0]
    print(f"First value in {column}: {first_value}")




#get matching frame/position

# Load the subtitles and video csv files to syncronise
file2_path = csv_file_path
file1_path = 'final_subtitles.csv'

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

