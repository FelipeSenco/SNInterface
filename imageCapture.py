import cv2

# Initialize the webcam. The argument can be the index of the webcam on your system.
# If you only have one webcam, it's normally at index 0.

cap = cv2.VideoCapture(0)

# Make sure the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
try:
    while True:
        # Read the current frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Display the resulting frame in a window named 'Webcam'
        cv2.imshow("Webcam", frame)

        print("Press 'q' to quit")
        # Wait for user input - if 'q' is pressed, break the loop
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
finally:
    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
