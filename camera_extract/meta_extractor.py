
import pyexiv2
import os

import utm
import numpy as np

import math
import random


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


folder_path = "/home/axelwagner/new_images"

drone_pos_txt = []

north, east = 0,0

for i in range(0, len(os.listdir(folder_path)) ):
    image_name = f"image_{i}.jpg"
    image_path = os.path.join(folder_path, image_name)
    
    if os.path.exists(image_path):
        tmp_list = []
        metadata = pyexiv2.ImageMetadata(image_path)
        metadata.read()

        xmp_dict = {}  # Create an empty dictionary to store the XMP key-value pairs

        # Iterate through the XMP keys and store the key-value pairs in the dictionary
        for key in metadata.xmp_keys:
            value = metadata[key].value
            if key in ["Xmp.xmp.CreateDate","Xmp.drone-dji.GpsLatitude","Xmp.drone-dji.GpsLongitude","Xmp.drone-dji.AbsoluteAltitude","Xmp.drone-dji.GimbalRollDegree",
                    "Xmp.drone-dji.GimbalYawDegree","Xmp.drone-dji.GimbalPitchDegree","Xmp.drone-dji.FlightRollDegree","Xmp.drone-dji.FlightYawDegree",
                    "Xmp.drone-dji.FlightPitchDegree"]:
                xmp_dict[key] = value

        image_name = os.path.basename(image_path)
        xmp_dict["file_name"] = image_name


        euler_Gimbal = [float(xmp_dict["Xmp.drone-dji.GimbalRollDegree"])*(math.pi/180),float(xmp_dict["Xmp.drone-dji.GimbalPitchDegree"])*(math.pi/180),float(xmp_dict["Xmp.drone-dji.GimbalYawDegree"])*(math.pi/180)]
        result_quaternion_gimbal = get_quaternion_from_euler(euler_Gimbal[0], euler_Gimbal[1], euler_Gimbal[2])
        height = float(xmp_dict["Xmp.drone-dji.GpsLatitude"])
        UTM_E , UTM_N, Zone, _ = utm.from_latlon(float(xmp_dict["Xmp.drone-dji.GpsLatitude"]), float(xmp_dict["Xmp.drone-dji.GpsLongitude"]) )
        if i == 0:
            north, east = UTM_N , UTM_E
        
        UTM_E = UTM_E - east
        UTM_N = UTM_N - north
        
        tmp_list = [UTM_N, UTM_E,height,result_quaternion_gimbal[0],result_quaternion_gimbal[1],result_quaternion_gimbal[2],result_quaternion_gimbal[3]]
    
        # ["height","x","y","qw","qx","qy","qz"]  
        drone_pos_txt.append(tmp_list)       

file_path = f'{folder_path}/{"opt_poses.txt"}'
with open(file_path, 'w') as file:
    for value_set in drone_pos_txt:
        line = " ".join(str(value) for value in value_set)
        file.write(line + '\n')


print("File 'data.txt' created successfully!")

import yaml

data ={
    "label": "sensorpod: calibration hitimo",
    "cameras": [
        {
            "camera": {
                "label": "cam0",
                "distortion": {
                    "parameters": {
                        "cols": 1,
                        "rows": 4,
                        "data": [0, 0, 0, 0]
                    },
                    "type": "equidistant"
                },
                "image_height": 1080,
                "image_width": 1920,
                "intrinsics": {
                    "cols": 1,
                    "rows": 4,
                    "data": [7389.595375722543, 7243.538461538462, 2640, 1973]
                },
                "type": "pinhole",
                "line-delay-nanoseconds": 0
            },
            "T_B_C": {
                "cols": 4,
                "rows": 4,
                "data": [1, 0, 0, 0,
                         0, 1, 0, 0,
                          0, 0, 1, 0,
                          0, 0, 0, 1]
            }
        }
    ]
}

yaml_file_path = f'{folder_path}/{"calibration.yaml"}'
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)




