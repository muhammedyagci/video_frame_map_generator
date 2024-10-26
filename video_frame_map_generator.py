import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, messagebox


# Specify the paths for the video and new image
video_path = 'video.mp4'
new_image_path = 'new_overlaypicture.jpeg'
mapping_path = 'frame_mappings.txt'  # Mapping file

# Load the image (loaded in BGR format, so we convert it to RGB)
new_overlay_image = cv2.imread(new_image_path)
new_overlay_image = cv2.cvtColor(new_overlay_image, cv2.COLOR_BGR2RGB)

# Resizing scale for preview and video
resize_scale = 0.5

# Load or create the mapping file (txt format)
frame_mappings = {}
comments = {}

try:
    with open(mapping_path, 'r') as f:
        for line in f:
            if '#' in line:
                start = line.find('#')
                comment = line[start:].strip()
                line = line[:start].strip()
            else:
                comment = ''

            if not line:
                continue

            frame_num, zoom, x_off, y_off = line.strip().split(',')
            frame_mappings[frame_num] = {
                'zoom_factor': float(zoom),
                'x_offset': int(x_off),
                'y_offset': int(y_off)
            }
            comments[frame_num] = comment
except FileNotFoundError:
    frame_mappings = {}
    comments = {}

# Define the zoom and position variables
zoom_factor = 1.0
x_offset, y_offset = 320, 872
frame_number = 0  # Current frame number
start_frame = None
end_frame = None

# Default sensitivity values
zoom_sensitivity = 0.02
position_sensitivity = 1

# Check if the settings menu is active or inactive
settings_menu_active = False
shortcuts_visible = False  # Visibility status of the shortcut menu

# Keyboard shortcuts
shortcuts = {
    'Zoom In': '+',
    'Zoom Out': '-',
    'Next Frame': 'n',
    'Previous Frame': 'p',
    'Move Up': 'w',
    'Move Down': 's',
    'Move Left': 'a',
    'Move Right': 'd',
    'Start Frame': 'b',
    'End Frame and Apply Interpolation': 'e',  # 'e' sets the end frame and applies interpolation
    'Increase Zoom Sensitivity': 'i',
    'Decrease Zoom Sensitivity': 'k',
    'Increase Position Sensitivity': 'j',
    'Decrease Position Sensitivity': 'l',
    'Apply Smoothing': 'm',  # 'm' for smoothing operation
    'Exit': 'q',
    'Show Shortcuts': 'h'
}


# Function for smoothing
def apply_smoothing():
    df = pd.read_csv(mapping_path, comment='#', names=["Frame", "Zoom", "PosX", "PosY"], delimiter=",")
    
    df['Zoom'] = df['Zoom'].rolling(window=4, min_periods=1).mean()
    df['PosX'] = df['PosX'].rolling(window=4, min_periods=1).mean()
    df['PosY'] = df['PosY'].rolling(window=4, min_periods=1).mean()

    df['PosX'] = df['PosX'].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df['PosY'] = df['PosY'].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    df['PosX'] = df['PosX'].astype(int)
    df['PosY'] = df['PosY'].astype(int)
    df['Zoom'] = df['Zoom'].round(3)

    df.to_csv(mapping_path, index=False, header=False)
    print("Smoothed data saved.")

# Confirmation window for smoothing
def confirm_smoothing():
    root = Tk()
    root.withdraw()  # Hide the Tkinter window
    result = messagebox.askyesno("Smoothing Process", "It's recommended to apply this process once at the end of all your work, otherwise there may be inconsistencies in the image. Do you want to proceed?")
    if result:
        apply_smoothing()
    else:
        print("Smoothing process canceled.")
    root.destroy()


# Function to display shortcuts in the terminal
def show_shortcut_menu_terminal():
    print("\n==== Shortcuts ====")
    for action, key in shortcuts.items():
        print(f"{action}: '{key}' key")
    print("====================\n")

# Function to display shortcuts in the OpenCV window
def draw_shortcuts(frame):
    y_offset = 30
    for action, key in shortcuts.items():
        cv2.putText(frame, f"{action}: {key}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

# Interpolation function: Applied between the selected two frames
def interpolate_mappings(start_frame, end_frame):
    start_mapping = frame_mappings.get(str(start_frame))
    end_mapping = frame_mappings.get(str(end_frame))

    if start_mapping is None or end_mapping is None:
        print("Make sure that the frames to be interpolated are recorded in the mapping.")
        return

    # For each frame where interpolation will be applied
    for frame in range(start_frame + 1, end_frame):
        ratio = (frame - start_frame) / (end_frame - start_frame)
        interpolated_zoom = start_mapping['zoom_factor'] * (1 - ratio) + end_mapping['zoom_factor'] * ratio
        interpolated_x = int(start_mapping['x_offset'] * (1 - ratio) + end_mapping['x_offset'] * ratio)
        interpolated_y = int(start_mapping['y_offset'] * (1 - ratio) + end_mapping['y_offset'] * ratio)

        # Fill each frame with interpolation, ignoring previous settings
        frame_mappings[str(frame)] = {
            'zoom_factor': interpolated_zoom,
            'x_offset': interpolated_x,
            'y_offset': interpolated_y
        }

# Function to save the mapping of a frame
def save_frame_mapping(frame_number):
    frame_mappings[str(frame_number)] = {
        'zoom_factor': zoom_factor,
        'x_offset': x_offset,
        'y_offset': y_offset
    }

    # Save the map to the TXT file without printing the comment lines
    with open(mapping_path, 'w') as f:
        for frame_num, mapping in frame_mappings.items():
            f.write(f"{frame_num},{mapping['zoom_factor']},{mapping['x_offset']},{mapping['y_offset']}\n")


# Function to apply interpolation and save the results to the file
def apply_interpolation():
    if start_frame is not None and end_frame is not None:
        interpolate_mappings(start_frame, end_frame)
        save_frame_mapping(end_frame)  # Save the end frame as well
        print("Interpolation applied")  # Indicate that interpolation is applied
    else:
        print("Please define the range of frames for interpolation.")

# Function to clear green areas and add the new image to the background
def process_frame(frame, frame_number):
    global zoom_factor, x_offset, y_offset

    if str(frame_number) in frame_mappings:
        mapping = frame_mappings[str(frame_number)]
        zoom_factor = mapping['zoom_factor']
        x_offset = mapping['x_offset']
        y_offset = mapping['y_offset']
    else:
        save_frame_mapping(frame_number)

    frame_copy = frame.copy()
    frame_height, frame_width, _ = frame_copy.shape

    # Convert the image to HSV format
    hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)

    # Define the range for green color
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    # Create the mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    frame_copy[mask != 0] = [0, 0, 0]  # Turn green areas black

    # Resize the new image based on the zoom factor
    new_size = int(400 * zoom_factor)

    # Resize the image
    resized_overlay = cv2.resize(new_overlay_image, (new_size, new_size))

    # Create a circular mask
    circle_mask = np.zeros((new_size, new_size), dtype=np.uint8)
    cv2.circle(circle_mask, (new_size // 2, new_size // 2), new_size // 2, 255, -1)

    # Create a 3-channel mask (for RGB)
    circular_overlay = np.zeros_like(resized_overlay)
    for i in range(3):
        circular_overlay[:, :, i] = np.where(circle_mask == 255, resized_overlay[:, :, i], 0)

    x_end = min(x_offset + new_size, frame_width)
    y_end = min(y_offset + new_size, frame_height)

    x_start = max(x_offset, 0)
    y_start = max(y_offset, 0)

    cropped_circular_overlay = circular_overlay[y_start - y_offset:(y_end - y_offset),
                                                x_start - x_offset:(x_end - x_offset)]

    # Place the RGB channels of the image
    for c in range(0, 3):
        frame_copy[y_start:y_end, x_start:x_end, c] = np.where(
            mask[y_start:y_end, x_start:x_end] == 0,
            frame_copy[y_start:y_end, x_start:x_end, c],
            cropped_circular_overlay[:, :, c]
        )

    resized_frame = cv2.resize(frame_copy, (int(frame_width * resize_scale), int(frame_height * resize_scale)))

    # Display the frame number on the screen
    cv2.putText(resized_frame, f"Frame: {frame_number}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # If the settings menu is active, display the sensitivity information on the screen
    if settings_menu_active:
        cv2.putText(resized_frame, f"Zoom Sensitivity: {zoom_sensitivity}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(resized_frame, f"Position Sensitivity: {position_sensitivity}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show shortcuts in the window
    if shortcuts_visible:
        draw_shortcuts(resized_frame)

    return resized_frame

# Function to display the current frame
def update_frame():
    global frame_number
    frame = frames[frame_number]
    processed_frame = process_frame(frame, frame_number)
    cv2.imshow('Processed Video', processed_frame)

# Load the video file and process the frames
def process_video():
    global zoom_factor, x_offset, y_offset, frame_number, zoom_sensitivity, position_sensitivity
    global start_frame, end_frame, shortcuts_visible, settings_menu_active  # Global variables

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    global frames
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    update_frame()  # Load the first frame

    while True:
        key = cv2.waitKey(0)

        if key == ord('+'):  # Zoom in
            zoom_factor += zoom_sensitivity
            save_frame_mapping(frame_number)
            update_frame()

        elif key == ord('-'):  # Zoom out
            zoom_factor = max(0.02, zoom_factor - zoom_sensitivity)
            save_frame_mapping(frame_number)
            update_frame()

        elif key == ord('n'):  # Move to the next frame
            frame_number += 1
            if frame_number >= frame_count:
                frame_number = frame_count - 1
            update_frame()

        elif key == ord('p'):  # Move to the previous frame
            frame_number = max(0, frame_number - 1)
            update_frame()

        elif key == ord('w'):  # Move up
            y_offset -= position_sensitivity
            save_frame_mapping(frame_number)
            update_frame()

        elif key == ord('s'):  # Move down
            y_offset += position_sensitivity
            save_frame_mapping(frame_number)
            update_frame()

        elif key == ord('a'):  # Move left
            x_offset -= position_sensitivity
            save_frame_mapping(frame_number)
            update_frame()

        elif key == ord('d'):  # Move right
            x_offset += position_sensitivity
            save_frame_mapping(frame_number)
            update_frame()

        # Increase zoom sensitivity
        elif key == ord('i'):  # Increase zoom sensitivity
            zoom_sensitivity += 0.01
            print(f"Zoom Sensitivity: {zoom_sensitivity}")
            update_frame()  # Display the change immediately

        # Decrease zoom sensitivity
        elif key == ord('k'):  # Decrease zoom sensitivity
            zoom_sensitivity = max(0.01, zoom_sensitivity - 0.01)
            print(f"Zoom Sensitivity: {zoom_sensitivity}")
            update_frame()  # Display the change immediately

        # Increase position sensitivity
        elif key == ord('j'):  # Increase position sensitivity
            position_sensitivity += 1
            print(f"Position Sensitivity: {position_sensitivity}")
            update_frame()  # Display the change immediately

        # Decrease position sensitivity
        elif key == ord('l'):  # Decrease position sensitivity
            position_sensitivity = max(1, position_sensitivity - 1)
            print(f"Position Sensitivity: {position_sensitivity}")
            update_frame()  # Display the change immediately

        # Set the start frame
        elif key == ord('b'):  # Select the start frame with the 'b' key
            start_frame = frame_number
            print(f"Start frame: {start_frame}")

        # Set the end frame and apply interpolation
        elif key == ord('e'):  # Select the end frame with the 'e' key and automatically apply interpolation
            end_frame = frame_number
            print(f"End frame: {end_frame}")
            apply_interpolation()  # Automatically apply interpolation
            update_frame()  # Show the results immediately

        # Open/close the settings menu
        elif key == ord('z'):
            settings_menu_active = not settings_menu_active
            print(f"Settings menu {'active' if settings_menu_active else 'inactive'}")
            update_frame()  # Show or hide the settings menu

        # Show/hide shortcuts
        elif key == ord('h'):
            show_shortcut_menu_terminal()  # Display in the terminal
            shortcuts_visible = not shortcuts_visible
            update_frame()  # Show or hide shortcuts on screen



        elif key == ord('m'):  # Shortcut for smoothing operation
            confirm_smoothing()
    

        elif key == ord('q'):  # Exit
            save_frame_mapping(frame_number)
            break

    cv2.destroyAllWindows()

# Start the video processing function
process_video()
