import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Define a mapping from class IDs to class names (modify as needed)
class_names = {
    0: 'unknown', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    # ... add more class names as needed
}

# Load the pre-trained model
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if frame was read successfully
    if not ret:
        print("Error reading frame from camera")
        break

    # Preprocess the frame for the model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))
    input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run object detection
    detector_output = detector.signatures['serving_default'](input_tensor)

    # Process the output
    boxes = detector_output['detection_boxes'].numpy()
    scores = detector_output['detection_scores'].numpy()
    classes = detector_output['detection_classes'].numpy()

    # Draw bounding boxes and labels on the frame
    for box, score, class_id in zip(boxes[0], scores[0], classes[0]):
        if score > 0.5:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get the class name from the mapping
            class_name = class_names.get(class_id, 'Unknown')

            # Draw the bounding box and label
            label = f"{class_name} ({score:.2f})"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_x = x1
            label_y = y1 - label_size[1] - baseline // 2
            cv2.rectangle(frame, (label_x, label_y), (label_x + label_size[0], label_y + label_size[1] + baseline), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (label_x, label_y + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
