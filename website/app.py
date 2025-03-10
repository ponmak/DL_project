import streamlit as st
import os
import ultralytics
import supervision as sv
import cv2
import torch
from detection import detection
from ultralytics import YOLO
from PIL import Image,ImageOps


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Title for the app
st.title("Plate Detection and Cost Calculation")
st.subheader("dev : ข้าวทุกจาน อาหารทุกอย่าง team")

# Load the YOLO model
model = YOLO("project/test_model/runs/detect/train/weights/best.pt")  # Load the trained model
#model.fuse()  # Fuse the model for improved inference speed

# choose to use camera or upload files
st.subheader("Choose to use camera or upload files")
camera = st.camera_input("Take a picture")
uploadfiles = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


# Check if the camera input is used
if camera:
    # Create a directory to save the images
    if not os.path.exists("images"):
        os.makedirs("images")

    # Save the image from the camera input
    file_path = os.path.join("images", "captured_image.jpg")
    with open(file_path, "wb") as f:
        f.write(camera.getbuffer())

    st.success("Image saved successfully!")

    # Display the captured image
    st.image(camera, caption="Captured Image", use_container_width=True)

    # Run detection on the captured image
    annotated_image = detection(file_path, model)
    # Save the annotated image
    annotated_image_path = os.path.join("images", "annotated_" + file.name)
    cv2.imwrite(annotated_image_path, annotated_image)

    # Display the annotated image
    st.image(annotated_image, caption="Annotated Image", use_container_width=True)


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
        # # flip the image
        # image_flip = Image.open(file)
        # image_flip = Image.rotate(image_flip, 180)  # Rotate the image by 180 degrees
        st.image(file.getvalue(), caption=file.name, use_container_width=True)  # Fix for Streamlit image handling

    # Run detection for each uploaded image
    for file in uploadfiles:
        file_path = os.path.join("images", file.name)
        annotated_image = detection(file_path, model)

        # Save the annotated image
        annotated_image_path = os.path.join("images", "annotated_" + file.name)
        cv2.imwrite(annotated_image_path, annotated_image)

        # Display the annotated image
        st.image(annotated_image, caption="Annotated Image", use_container_width=True)
      
    