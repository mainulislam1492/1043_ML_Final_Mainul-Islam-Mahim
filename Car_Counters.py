import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Added this import

# --- Configuration ---
VIDEO_SOURCE    = 'cars.mp4'
MIN_WIDTH       = 40
MIN_HEIGHT      = 40
OFFSET          = 10
LINE_HEIGHT     = 550
LINE_END_X      = 1200
RECT_PAD        = 10

# --- Setup ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FPS, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

matches = []
car_count = 0

def get_centroid(x, y, w, h):
    return (x + w // 2, y + h // 2)

# --- Read first two frames ---
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# --- Main Loop ---
while ret:
    diff    = cv2.absdiff(frame1, frame2)
    grey    = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(grey, (5, 5), 0)
    _, thresh   = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated     = cv2.dilate(thresh, np.ones((3, 3)))

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (0, LINE_HEIGHT), (LINE_END_X, LINE_HEIGHT), (0, 255, 0), 2)

    for _, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)

        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        cx, cy = get_centroid(x, y, w, h)
        matches.append((cx, cy))

        cv2.rectangle(frame1, (x - RECT_PAD, y - RECT_PAD), (x + w + RECT_PAD, y + h + RECT_PAD), (255, 0, 0), 2)
        cv2.circle(frame1, (cx, cy), 5, (0, 255, 0), -1)

    # --- Count vehicles crossing the line ---
    for pt in matches[:]:
        if (LINE_HEIGHT - OFFSET) < pt[1] < (LINE_HEIGHT + OFFSET):
            car_count += 1
            matches.remove(pt)
            print(f"Vehicle Count: {car_count}")

    cv2.putText(frame1, f"Total Vehicles Detected: {car_count}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    cv2_imshow(frame1) # Changed cv2.imshow to cv2_imshow

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

    frame1 = frame2
    ret, frame2 = cap.read()

cap.release()
cv2.destroyAllWindows()
