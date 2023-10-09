import cv2

# Input video file and output video file paths
input_video_path = '/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/001-nm-03-162.avi'
input_video_path = '/home/c3-0/datasets/ID-Dataset/casiab/metadata/silhouettes/person/001/nm-03/162.mp4'
output_video_path = '/home/prudvik/id-dataset/dataset-augmentation/outputs/indices-debug/person_sil_trimmed.mp4'

# List of frame indices to keep
desired_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))

# Define the codec and create VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

frame_index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop when we reach the end of the video

    if frame_index in desired_indices:
        out.write(frame)  # Write the frame to the output video

    frame_index += 1

# Release video objects
cap.release()
out.release()

print("Video processing complete.")
