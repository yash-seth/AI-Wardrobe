# v1 - segmenting the various pieces of clothing

import cv2
import numpy as np
from fashn_human_parser import FashnHumanParser, IDS_TO_LABELS

# tasks:
# - how to single out each identified entity in the image such as t-shirt, glasses etc

# 1. Initialize parser and load image
parser = FashnHumanParser()
image_path = "./Images/rachkonda_sparsh.JPG"

original_bgr = cv2.imread(image_path)
h, w, _ = original_bgr.shape

# 2. Get the 2D segmentation map (shape: H, W)
mask_2d = parser.predict(image_path)
print(mask_2d)

# Ensure dimensions match perfectly
if mask_2d.shape[:2] != (h, w):
    mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)

# 3. Create a unique, distinct color palette for the 18 classes
np.random.seed(42)  # Fixed seed for consistent mask colors
colors = np.random.randint(0, 255, size=(18, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # Force background (ID 0) to stay black

# 4. Map the 2D segmentation IDs to the 3D BGR color palette
colored_mask = colors[mask_2d]

# 5. Blend the original image and the colored mask (Alpha Blending)
alpha = 0.6
beta = 0.4
overlay_result = cv2.addWeighted(original_bgr, alpha, colored_mask, beta, 0)

# 6. Find masks and dynamically draw text labels onto the segments
# Loop over every possible label ID except 0 (Background)
for class_id in range(1, 18):
    # Create a binary mask specifically for this class
    class_mask = (mask_2d == class_id).astype(np.uint8)
    
    # Check if the class is actually present in the image
    if np.any(class_mask):
        # Find spatial moments of the class mask to locate its center
        moments = cv2.moments(class_mask)
        
        if moments["m00"] > 0:  # Avoid division by zero
            # Calculate coordinates for the center of the segment
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Retrieve string name from fashn mapping utilities
            label_text = IDS_TO_LABELS.get(class_id, f"ID {class_id}")
            
            # Setup text parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Draw a black text shadow/outline for visibility against any color
            cv2.putText(overlay_result, label_text, (cx + 1, cy + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            # Draw the clean white label text in the center
            cv2.putText(overlay_result, label_text, (cx, cy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# 7. Display and save the labeled result
cv2.imshow("Human Parser with Labels", overlay_result)
cv2.imwrite("labelled_" + image_path.split('/')[-1], overlay_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

