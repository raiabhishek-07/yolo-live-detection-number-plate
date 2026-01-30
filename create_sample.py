import cv2
import numpy as np
import random

# Create a background with some noise (gray-ish)
img = np.zeros((400, 600, 3), dtype=np.uint8) + 200
noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
img = cv2.add(img, noise)

# Create a plate region
# Size: 250x70 (Ratio ~3.57, Area 17500)
x, y, w, h = 150, 165, 250, 70

# Draw yellow plate background
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), -1)

# Add Border
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)

# Add Text "MH12AB1234"
# Font scale 1.0, Thickness 3
font = cv2.FONT_HERSHEY_SIMPLEX
text = "MH12AB1234"
(text_w, text_h), _ = cv2.getTextSize(text, font, 1.0, 3)

# Center text
tx = x + (w - text_w) // 2
ty = y + (h + text_h) // 2
cv2.putText(img, text, (tx, ty), font, 1.0, (0, 0, 0), 3)

# Add screw holes (for edge density)
cv2.circle(img, (x+15, y+35), 3, (50, 50, 50), -1)
cv2.circle(img, (x+w-15, y+35), 3, (50, 50, 50), -1)

# Save
cv2.imwrite("sample_plate.jpg", img)
print("Created improved sample_plate.jpg")
