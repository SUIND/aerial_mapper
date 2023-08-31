# import cv2
# import pytesseract

# # Set the path to the video file
# video_path = '/home/axelwagner/Downloads/test_1/DJI_0354.MP4'
# #pytesseract.pytesseract.tesseract_cmd = '/usr/share/tesseract-ocr'  # Set the path to your Tesseract executable

# # Open the video
# cap = cv2.VideoCapture(video_path)

# # Get the original frame dimensions
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # Define the new dimensions for the resized frames
# new_width = 640  # Set your desired width
# new_height = 480  # Set your desired height

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     print(1)
#     # Resize the frame to the new dimensions
#     resized_frame = cv2.resize(frame, (new_width, new_height))

#     # Convert the resized frame from BGR to RGB format
#     frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

#     # Apply OCR using pytesseract
#     subtitles = pytesseract.image_to_string(frame_rgb)

#     # Print the extracted subtitles
#     if subtitles.strip():  # Only print non-empty subtitles
#         print(subtitles)

#     # Display the resized frame
#     cv2.imshow('Resized Frame', resized_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import subprocess

# # Set the path to the video file
# video_path = '/home/axelwagner/Downloads/test_1/DJI_0354.MP4'

# # Run FFmpeg command to extract subtitles
# ffmpeg_command = [
#     'ffmpeg',
#     '-i', video_path,
#     '-f', 'csv',  # Specify the output format as SubRip (srt)
#     '-vn', '-an',  # Disable video and audio streams
#     '-map', '0:s:0',  # Select the first subtitle stream
#     'output.srt'  # Output subtitle file
# ]

# subprocess.run(ffmpeg_command)

# import re
# import csv

# # Read the srt file and split it into subtitle entries
# with open('output.srt', 'r', encoding='utf-8') as srt_file:
#     srt_content = srt_file.read()

# # Split the srt content into individual subtitle entries
# subtitle_entries = re.split(r'\n\n', srt_content)

# # Initialize a CSV writer
# csv_file = open('subtitles.csv', 'w', newline='', encoding='utf-8')
# csv_writer = csv.writer(csv_file)

# # Write CSV header
# csv_writer.writerow(['Subtitle Number', 'Start Time', 'End Time', 'Subtitle Text'])

# # Process each subtitle entry
# for i, entry in enumerate(subtitle_entries):
#     lines = entry.strip().split('\n')
#     if len(lines) >= 3 and re.match(r'^\d+$', lines[0]):
#         start_time, end_time = re.findall(r'(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
#         subtitle_text = '\n'.join(lines[2:])
#         csv_writer.writerow([i + 1, start_time, end_time, subtitle_text])

# # Close the CSV file
# csv_file.close()

import pandas as pd
import re
# Read the CSV file into a DataFrame
csv_file_path = 'subtitles.csv'
df = pd.read_csv(csv_file_path)

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

keys_to_extract = ['F','SS', 'ISO', 'EV', 'DZOOM', 'GPS', 'D', 'H', 'H.S', 'V.S']

for key in keys_to_extract:
    df[key] = df['Subtitle Text'].apply(lambda x: extract_value(x, key))

# Drop the "Subtitle Text" column
df.drop(columns=['Subtitle Text'], inplace=True)

# Save the modified DataFrame back to a CSV file
output_csv_file_path = 'new_subtitles.csv'
df.to_csv(output_csv_file_path, index=False)



# Print the first value for each column
for column in df.columns:
    first_value = df[column].iloc[0]
    print(f"First value in {column}: {first_value}")
