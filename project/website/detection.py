from ultralytics import YOLO
import supervision as sv
import torch
import cv2

# Detection function
def detection(image, model):
    # Perform inference
    results = model(image)
    detections = sv.Detections.from_ultralytics(results[0]).with_nms()  # Access the first element of the results list

    return detections

# Annotation function
def annotation(detections, image_path): 
    # Create an empty list to store annotations
    annotations = []

    # Iterate through the detections
    for i in range(len(detections.boxes)):
        # Get the box coordinates and class name
        x1, y1, x2, y2 = detections.xyxy[i]
        class_name = detections.names[detections.boxes.cls[i].item()]  # Convert tensor to item
        confidence = detections.boxes.conf[i].item()  # Convert tensor to item

        # Append the annotation to the list
        annotations.append((x1, y1, x2, y2, f"{class_name} {confidence:.2f}"))
        # Draw the bounding box and label on the image
        cv2.rectangle(image_path, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_path, f"{class_name} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the image to a format suitable for display
    image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    # Save the annotated image
    annotated_image_path = image_path.replace(".jpg", "_annotated.jpg")
    cv2.imwrite(annotated_image_path, image_path)
    

    return annotations, annotated_image_path

# Display function using supervision
def display(annotations, image_path):
    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Create a canvas for drawing
    canvas = sv.Canvas(image=image)

    # Draw the annotations on the canvas
    for ann in annotations:
        x1, y1, x2, y2, label = ann
        canvas.draw_rectangle(top_left=(int(x1), int(y1)), bottom_right=(int(x2), int(y2)), color=sv.Color.GREEN)
        canvas.draw_text(text=label, position=(int(x1), int(y1) - 10), color=sv.Color.GREEN)

    # Convert the canvas back to an image
    annotated_image = canvas.get_image()

    return annotated_image