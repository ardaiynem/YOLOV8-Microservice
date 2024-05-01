from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__, template_folder='template')

# Load the ONNX model
model_path = './yolov8m.onnx'
model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Classes that pretrained YOLOv8 model can predicted are added hardcoded for the sake of simplicity
yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

@app.route("/")
def root():
    """Root endpoint"""
    return render_template('index.html')

@app.route("/detect/<label>", methods=["POST"])
@app.route("/detect/", methods=["POST"])
def detect(label=None):
    """
    Endpoint for object detection.
    
    Args:
        None
        
    Returns:
        JSON response containing detected objects information.
    """
    # Get the request paramaters, namely the file and the label
    file = request.files["file"]
    # label = request.form.get('label')

    # If label is provided in the URL path, use it, otherwise get it from the form data
    if label is None:
        label = request.form.get('label')
        return('Error, what is the label?')

    print(label)
    print(file)

    # Detect the objects
    image_content = file.stream.read()
    objects = detect_objects(file.stream, label)

    # Plot bounding boxes on the original image
    img = Image.open(io.BytesIO(image_content))
    plotted_img = plot_boxes_on_image(img, objects)

    # Convert the plotted image to UTF-8 format
    buffered = io.BytesIO()
    plotted_img.save(buffered, format="PNG")
    response_data = {
        "image": base64.b64encode(buffered.getvalue()).decode("utf-8"),
        "objects": objects,
        "count": len(objects),
    }

    return jsonify(response_data)

def plot_boxes_on_image(image, objects):
    """
    Plot bounding boxes on the original image.
    
    Args:
        image (PIL.Image): Original image.
        objects (list): Detected objects information.
        
    Returns:
        PIL.Image: Image with bounding boxes plotted.
    """
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot each bounding box
    for obj in objects:
        x, y, width, height = obj['x'], obj['y'], obj['width'], obj['height']
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Convert plot to image
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.draw()
    plt.close()
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img = Image.open(img_buffer)
    return img
    
def detect_objects(buf, label):
    """
    Detect objects on the provided image.
    
    Args:
        buf (file stream): Image file stream.
        label (str): Label of object to detect.
        
    Returns:
        List: Detected objects information.
    """
    # Prepate the input then run the model on the processed input data
    input, img_width, img_height = prepare_input(buf)
    output = run_model(input)
    result = process_output(output, img_width, img_height, label)

    return result

def prepare_input(buf):
    """
    Prepare the image for input to the model.
    
    Args:
        buf (file stream): Image file stream.
        
    Returns:
        tuple: Image input, width, and height.
    """
    img = Image.open(buf)
    img_width, img_height = img.size

    # Resizing image for YOLOv8 model that accepts 640 x 640 image inputs in RGB format
    img = img.resize((640, 640))
    img = img.convert("RGB")

    # Changing array dimensions from (640, 640, 3) to (3,640,640) for YOLO model
    input = np.array(img)
    input = input.transpose(2, 0, 1)

    # Adding one more shape to obtain (1,3,640,640) shape finally, since YOLO model accepts input of this shape
    # Also, array is normalized (Min-Max scaled) to be witin range [0 - 1]
    input = input.reshape(1, 3, 640, 640) / 255.0

    return input.astype(np.float32), img_width, img_height

def run_model(input):
    """
    Run the object detection model.
    
    Args:
        input (numpy.array): Model input.
        
    Returns:
        numpy.array: Model output.
    """

    # Make the predictions
    outputs = model.run(["output0"], {"images":input})

    # Returning the result with outputs[0], as only one image tensor has been provided as input. Output has shape (1, 84, 8400)
    return outputs[0]

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (dict): Bounding box 1.
        box2 (dict): Bounding box 2.
        
    Returns:
        float: Intersection over Union value.
    """
    return intersection(box1, box2) / union(box1, box2)

def union(box1, box2):
    """
    Calculate the union area of two bounding boxes.
    
    Args:
        box1 (dict): Bounding box 1.
        box2 (dict): Bounding box 2.
        
    Returns:
        float: Union area.
    """
    # Area of two input boxes are calculated
    box1_width, box1_height = box1["width"], box1["height"]
    box2_width, box2_height = box2["width"], box2["height"]
    box1_area = box1_width * box1_height
    box2_area = box2_width * box2_height
    
    # Area of union is calculated
    area_union = box1_area + box2_area - intersection(box1, box2)
    return area_union

def intersection(box1, box2):
    """
    Calculate the intersection area of two bounding boxes.
    
    Args:
        box1 (dict): Bounding box 1.
        box2 (dict): Bounding box 2.
        
    Returns:
        float: Intersection area.
    """
    # x1, x2, y1 and y2 points for the intersection is calculated
    x1_intersection = max(box1["x"], box2["x"])
    y1_intersection = max(box1["y"], box2["y"])
    x2_intersection = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2_intersection = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

    # Intersection area is calculated
    width_intersection = x2_intersection - x1_intersection
    height_intersection = y2_intersection - y1_intersection
    area_intersection = max(0, width_intersection) * max(0, height_intersection)

    return area_intersection

def process_output(output, img_width, img_height, label_requested):
    """
    Process the model output and return detected objects.
    
    Args:
        output (numpy.array): Model output.
        img_width (int): Image width.
        img_height (int): Image height.
        label_requested (str): Requested object label.
        
    Returns:
        List: Detected objects information.
    """

    # output has shape (1, 84, 8400), thus output[0] is used for eliminating redundant dimension, resulting in shape (84, 8400)
    output = output[0].astype(float)

    # Transposed version has shape (8400, 84), 8400 is a maximum number of bounding boxes that the YOLOv8 model can detect
    # Each row has 84 parameters, first 4 elements are coordinates of the bounding box and rest are the probabilities of all object classes that this model can detect. 
    output = output.transpose()

    boxes = []
    for row in output:
        # Extracting the most probable object predicted for the bounded box
        prob = row[4:].max()

        threshold_confidence = 0.5
        # Bounding boxes with low confidence are eliminated
        if prob < threshold_confidence:
            continue

        # Index of predicted class is extracted, then the label is extracted too
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]

        # x_center, y_center, width and height parameters of current bounding box is extracted
        xc, yc, w, h = row[:4]

        # Since original image has been scaled to (640, 640), parameters scaled according to original image size
        x1 = (xc - w/2) / 640 * img_width
        y1 = (yc - h/2) / 640 * img_height
        x2 = (xc + w/2) / 640 * img_width
        y2 = (yc + h/2) / 640 * img_height

        # Predictions which are not matching with requested label are filtered out
        if (label == label_requested):
            boxes.append({"label": label, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "confidence": prob})

    # Predictions are sorted in descending order
    boxes.sort(key=lambda x: x['confidence'], reverse=True)

    # Non-maximum suppression is applied, boxes with IoU rate more than 0.7 are eliminated
    non_max_suppression_constant = 0.7
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < non_max_suppression_constant]

    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)