import cv2
import numpy as np
def nothing(x):
    pass

# Video input
video_path = "people.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create Trackbars
cv2.namedWindow('HSV Trackbars')
cv2.createTrackbar("Low-H", "HSV Trackbars", 0, 0, nothing)
cv2.createTrackbar("High-H", "HSV Trackbars", 0, 255, nothing)
cv2.createTrackbar("Low-S", "HSV Trackbars", 0, 0, nothing)
cv2.createTrackbar("High-S", "HSV Trackbars", 50, 255, nothing)
cv2.createTrackbar("Low-V", "HSV Trackbars", 0, 0, nothing)
cv2.createTrackbar("High-V", "HSV Trackbars", 180, 255, nothing)


while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video, looping...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_h = cv2.getTrackbarPos("Low-H", "HSV Trackbars")
    high_h = cv2.getTrackbarPos("High-H", "HSV Trackbars")
    low_s = cv2.getTrackbarPos("Low-S", "HSV Trackbars")
    high_s = cv2.getTrackbarPos("High-S", "HSV Trackbars")
    low_v = cv2.getTrackbarPos("Low-V", "HSV Trackbars")
    high_v = cv2.getTrackbarPos("High-V", "HSV Trackbars")

    lower_bound = np.array([low_h, low_s, low_v])
    upper_bound = np.array([high_h, high_s, high_v])

    # create mask for the hsv frame
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 255
    # Filter by Area
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 10000

    params.filterByColor = False

    # Filter by Circularity
    params.filterByCircularity = 1
    params.minCircularity = 0.1
    params.maxCircularity = 1
    # Filter by Convexity
    params.filterByConvexity = 1
    params.minConvexity = 0.1
    params.maxConvexity = 1
    # Filter by Inertia
    params.filterByInertia = 1
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 0.9

# Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)

# print the frame which detects people
    for keypoint in keypoints:
        blob_frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Blob Detection', blob_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
