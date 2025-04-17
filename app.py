from flask import Flask, render_template, request
from flask_socketio import SocketIO
import base64
import io
from PIL import Image
import torch
from MSFCN2D import MSFCN2D
import numpy as np
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSFCN2D(time_num=4, band_num=4, class_num=4).to(device)

# Load the trained model weights
try:
    model.load_state_dict(torch.load("MSFCN2D_model.pth", map_location=device))
    model.eval()
except:
    print("Warning: Could not load model weights. Using untrained model.")

def process_image_for_model(image):
    # Convert PIL Image to tensor and preprocess
    # Resize to match model's expected input size
    image = image.resize((64, 64))
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.array(image)).float()
    
    # Reshape to match model's expected input shape
    # Assuming the model expects [batch_size, channels*time_num, height, width]
    if len(image_tensor.shape) == 3:  # RGB image
        image_tensor = image_tensor.permute(2, 0, 1)  # [C, H, W]
        # Repeat the channels to match time_num * band_num
        image_tensor = image_tensor.repeat(4, 1, 1)  # Adjust multiplier as needed
    
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

def generate_segmentation_mask(output):
    # Convert model output to segmentation mask
    # Assuming output shape is [batch_size, num_classes, height, width]
    pred = torch.argmax(output, dim=1)
    pred = pred.cpu().numpy()[0]  # Take first batch item
    
    # Create RGB mask
    colors = {
        0: [255, 0, 0],    # Urban (Red)
        1: [0, 255, 0],    # Vegetation (Green)
        2: [0, 0, 255],    # Water (Blue)
        3: [255, 255, 0]   # Bare Soil (Yellow)
    }
    
    rgb_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_idx, color in colors.items():
        rgb_mask[pred == class_idx] = color
    
    return Image.fromarray(rgb_mask)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('process_image')
def handle_image(image_data):
    try:
        # Remove the data URL prefix to get the base64 string
        image_data = image_data.split(',')[1]
        
        # Convert base64 to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Record start time
        start_time = time.time()
        
        # Process image
        input_tensor = process_image_for_model(image)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Generate segmentation mask
        segmented_image = generate_segmentation_mask(output)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
        
        # Calculate confidence score (example implementation)
        confidence_score = float(torch.nn.functional.softmax(output, dim=1).max().item() * 100)
        
        # Convert segmented image to base64
        buffered = io.BytesIO()
        segmented_image = segmented_image.resize(image.size)  # Resize back to original size
        segmented_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Send results back to client
        socketio.emit('segmentation_result', {
            'image': f'data:image/png;base64,{img_str}',
            'processing_time': processing_time,
            'confidence': round(confidence_score, 2)
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        socketio.emit('error', {'message': 'Error processing image'})

if __name__ == '__main__':
    socketio.run(app, debug=True)