import supervision as sv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import gradio as gr
from PIL import Image

# Load the Detectron2 model
def get_model(config_path: str, weights_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    predictor = DefaultPredictor(cfg)
    return predictor

# Path to the config and weights files
config_path = "/teamspace/studios/this_studio/config.yml"
weights_path = "/teamspace/studios/this_studio/output/model_final.pth"
model = get_model(config_path, weights_path)

# Define your polygonal ROI coordinates
#polygon = [(175, 628), (515, 330), (1305, 330), (1624, 640), (1648, 1949), (1300, 2270), (500, 2260), (160, 1940)]
polygon = [(340, 1020),(840,530),(2070,530),(2540, 1000), (2515, 2962), (2030, 3440), (800, 3458), (320, 2985)]

# Create ROI mask
def create_roi_mask(image_shape, polygon):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask

# Filtering large bounding boxes
def filter_large_bounding_boxes(detections, max_area):
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area <= max_area:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_classes.append(class_id)

    if not filtered_boxes:
        return sv.Detections.empty()

    filtered_boxes = np.array(filtered_boxes)
    filtered_scores = np.array(filtered_scores)
    filtered_classes = np.array(filtered_classes)

    detection_data = {
        "xyxy": filtered_boxes,
        "confidence": filtered_scores,
        "class_id": filtered_classes
    }
    return sv.Detections(**detection_data)

# Shrimp detection function
def detect_shrimp(image):
    image_np = np.array(image)
    image = cv2.imread(image_path)
    original_dims = (3000,4000)
    image_np = cv2.resize(image, (original_dims[0], original_dims[1]))
    mask = create_roi_mask(image_np.shape, polygon)
    masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)
    
    slicer = sv.InferenceSlicer(
        callback=slicer_callback,
        slice_wh=(520, 520),  # Adjusted slice size
        overlap_ratio_wh=(0.7, 0.7),
        overlap_filter_strategy=sv.OverlapFilter.NON_MAX_MERGE
    )
    detections = slicer(masked_image)

    # Filter large bounding boxes
    max_area = 900  # Define the maximum allowed area for bounding boxes (example: 900 pixels)
    filtered_detections = filter_large_bounding_boxes(detections, max_area)

    # Count the number of bounding boxes
    num_filtered_detections = len(filtered_detections.xyxy)
    print(f"Number of filtered bounding boxes: {num_filtered_detections}")

    # Annotate the original image with filtered detections
    annotator = sv.BoundingBoxAnnotator()
    annotated_frame_filtered = annotator.annotate(scene=image_np.copy(), detections=filtered_detections)

    annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame_filtered, cv2.COLOR_BGR2RGB))
    return annotated_image, num_filtered_detections

def slicer_callback(slice: np.ndarray) -> sv.Detections:
    outputs = model(slice)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.array([])
    scores = instances.scores.numpy() if instances.has("scores") else np.array([])
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.array([])

    if boxes.shape[0] == 0:  # No detections
        return sv.Detections.empty()

    # Convert the outputs to a compatible format for sv.Detections
    detection_data = {
        "xyxy": boxes,
        "confidence": scores,
        "class_id": classes
    }
    detections = sv.Detections(**detection_data)
    return detections

def process_input(source, image):
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, "Failed to capture image", ""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    elif image is not None:
        image = Image.fromarray(image)
    else:
        return None, "No image provided", ""
    
    annotated_image, count = detect_shrimp(image)
    return annotated_image, count, ""

# Define the interface
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Radio(choices=["Webcam", "Upload"], label="Select Image Source"),
        gr.Image(type="numpy", label="Upload an Image or Capture from Webcam")
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated Image"),
        gr.Textbox(label="Number of Shrimps Detected")
    ],
    live=True,
    title="Shrimp Counter",
    description="Detect shrimp in images or from webcam.",
    allow_flagging="never"  # Disable the flag button
)

iface.launch(share=True)
