
# Visual Mapping Extraction

This project allows determining the coordinates and zoom index of an image to be overlaid on a video, on a frame-by-frame basis, and saving this data into a mapping file. The mapping file contains zoom and position information to control the image's placement and size in the video. **We will use this mapping later in our video processing application.**

## Features

1. **Frame-by-Frame Zoom and Position Adjustment:**  
   The zoom level and position of the image overlay can be manually adjusted for each frame. These settings are saved separately for each frame and can later be filled in using interpolation.

2. **Keyboard Shortcuts for Easy Control:**  
   You can easily navigate through frames, modify zoom levels, and adjust the position of the image using keyboard shortcuts. These settings are then saved in the mapping file.

3. **Smooth Transitions with Interpolation:**  
   Zoom and position values between two specified frames can be automatically calculated using interpolation, creating smooth transitions between frames.

4. **Green Area Masking:**  
   Green areas in the video are masked, and the new image is placed behind these areas.

5. **Smoothing Feature:**  
   The final settings can be smoothed by applying a moving average to ensure smoother zoom and position transitions.

## Requirements

1. **Python Libraries:**
   - OpenCV (`cv2`)
   - NumPy (`numpy`)
   - Pandas (`pandas`)
   - Tkinter (`tkinter`) - for confirmation dialogs

2. **Output File:**
   - The `frame_mappings.txt` file is an output that contains the zoom and position settings for each frame. This file is generated during the project and used later in video processing applications.

---

## Explanation of Values in `frame_mappings.txt`

The `frame_mappings.txt` file contains information for each frame regarding the zoom factor and position offsets (x and y coordinates) of the image to be overlaid.

Each line in the file follows this format:

```
<frame_number>,<zoom_factor>,<x_offset>,<y_offset>
```

### Explanation of Fields:

- **frame_number:**  
  This is the index of the video frame for which the settings are defined. Each frame in the video is assigned a number starting from 1.

- **zoom_factor:**  
  This value controls the size of the overlay image. A `zoom_factor` of `1.0` means the image retains its original size, while values greater than `1.0` increase the size of the image, and values less than `1.0` reduce the size.

- **x_offset:**  
  This is the horizontal position offset of the overlay image, measured from the left side of the frame. A higher value shifts the image further to the right.

- **y_offset:**  
  This is the vertical position offset of the overlay image, measured from the top of the frame. A higher value shifts the image further down.

---

### Example Mapping File (`frame_mappings.txt`)

```
1,1.0,320,872
2,1.05,325,875
# This line is a comment
10,1.2,350,890
```

### Example Breakdown:

- **Frame 1:**  
  - Zoom factor: `1.0` (image retains its original size)
  - X offset: `320` (the image is positioned 320 pixels from the left)
  - Y offset: `872` (the image is positioned 872 pixels from the top)

- **Frame 2:**  
  - Zoom factor: `1.05` (the image is slightly zoomed in)
  - X offset: `325` (the image is shifted 5 pixels to the right compared to Frame 1)
  - Y offset: `875` (the image is shifted 3 pixels down compared to Frame 1)

- **Frame 10:**  
  - Zoom factor: `1.2` (the image is further zoomed in)
  - X offset: `350` (the image is shifted 30 pixels to the right compared to Frame 1)
  - Y offset: `890` (the image is shifted 18 pixels down compared to Frame 1)

---

### Usage Steps

### Step 1: Load Video and Image
- Place your video and image files in the project directory, and define the `video_path` and `new_image_path` variables.

### Step 2: Frame-by-Frame Adjustments
- Navigate through the video frames and adjust the zoom and position of the overlay image for each frame. 
- These settings are manually controlled via keyboard shortcuts and saved in the `frame_mappings.txt` file.

### Step 3: Navigating and Adjusting with Shortcuts
- Use the following keyboard shortcuts to adjust the size and position of the overlay image for each frame:
  - `n`: Move to the next frame
  - `p`: Move to the previous frame
  - `w/a/s/d`: Move the overlay image up/left/down/right
  - `+/-`: Zoom in/out
  - `b`: Set the start frame
  - `e`: Set the end frame and apply interpolation
  - `h`: Show the list of shortcuts
  - `q`: Quit and save settings

### Step 4: Create the Mapping File
- The settings for each frame are saved in the `frame_mappings.txt` file. **This mapping file is created for use in video processing applications later.**

### Step 5: Applying Interpolation
- Once the start and end frames are selected, interpolation automatically calculates and fills in the zoom and position values between the two frames.

### Step 6: Applying Smoothing (Optional)
- Press `m` to apply smoothing to all the settings, ensuring smoother transitions between frames.

### Step 7: Save and Exit
- Press `q` to save the current settings and exit the tool.

---

### Known Limitations
- Smoothing should be applied after all adjustments are completed. Applying it multiple times during the adjustment process may cause inconsistencies.

---

This tool allows precise adjustment of the zoom and position settings for each frame, saving them in a mapping file for later use in video processing tasks.
