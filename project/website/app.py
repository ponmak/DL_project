import streamlit as st
import os
import ultralytics
import supervision as sv
import cv2
from detection import detection, annotation, display  
from ultralytics import YOLO

# Title for the app
st.title("Plate Detection and Cost Calculation")
st.subheader("dev : ข้าวทุกจาน อาหารทุกอย่าง team")

# Get images file
uploadfiles = st.file_uploader(label="Upload Image", type=["jpg", "jpeg", "png"], key="image_file", accept_multiple_files=True)

if uploadfiles:
    # Create a directory to save the images
    if not os.path.exists("images"):
        os.makedirs("images")

    # Save the images to the directory
    for file in uploadfiles:
        file_path = os.path.join("images", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.success("Images saved successfully!")

    # Display the uploaded images
    for file in uploadfiles:
        st.image(file.getvalue(), caption=file.name, use_container_width=True)  # Fix for Streamlit image handling

    # Load the YOLO model
    model = YOLO("project/test_model/runs/detect/train/weights/best.pt")  # Load the trained model
    model.fuse()  # Fuse the model for improved inference speed

    # Run detection for each uploaded image
    for file in uploadfiles:
        st.write(f"Detecting {file.name}...")

        # Load the image
        image_path = os.path.join("images", file.name)
        image = cv2.imread(image_path)

        # Run detection
        results = detection(image, model)

        # Annotate the image
        annotated_image = annotation(results, image_path)
        

        # Display the annotated image
        annotated_image = display(annotations, image_path)
        st.image(annotated_image, caption=f"Annotated {file.name}", use_container_width=True)

        # Display detection results
        st.write("Results:")
        st.write(f"Number of plates detected: {len(results.boxes)}")
        st.write("Annotations:")
        for ann in annotations:
            st.write(f"Plate: {ann[4].split()[0] if len(ann[4].split()) > 0 else 'Unknown'}, Confidence: {ann[4].split()[1] if len(ann[4].split()) > 1 else 'Unknown'}")  # Fix index error and handle missing class names

        # Display the YOLO detection plot
        st.image(results.plot(), caption=f"Detection Plot {file.name}", use_container_width=True)
        st.success("Detection completed!")