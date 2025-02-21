# shrimp Seed Detection and Counting

## Overview
This project implements an automated system for detecting and counting prawn seeds/larvae using computer vision techniques. The system provides accurate, real-time counting to help shrimp hatchery operators and farmers monitor population density, improve inventory management, and optimize feeding regimes.

## Features
- **Automated Detection**: Identifies individual prawn seeds in images or video streams
- **Accurate Counting**: Provides precise counts of prawn seeds with minimal margin of error
- **Batch Processing**: Handles multiple images for large-scale analysis
- **User-Friendly Interface**: Simple controls for image upload and result visualization
- **Report Generation**: Creates detailed reports with count statistics

## Technology Stack
- **Programming Language**: Python 3.11
- **Computer Vision**: OpenCV, TensorFlow/PyTorch, detectron2 (from facebook)
- **Image Processing**: NumPy, SciPy
- **Model Architecture**: YOLOv5/Faster R-CNN for object detection
- **UI Framework**: Streamlit/Flask (web interface)

## Installation

### Prerequisites
- Python 3.11 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Webcam or digital camera for live counting

### Setup
```bash
# Clone the repository
git clone https://github.com/rishendra-manne/shrimp_seeds_detection.git
cd shrimp_seeds_detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### Web Interface
```bash
# Start the web application
python app.py
```
Then open your browser and navigate to `http://localhost:8501`

## Model Training

If you need to train the model on your specific prawn species or environment:

```bash
# Train the detection model
python src/pipelines/training_pipeline.py
```

## Performance Metrics
- **Counting Accuracy**: 93-96% under optimal conditions
- **Processing Speed**: 2-3 seconds for a image
- **Minimum Detectable Size**: 0.5mm larvae
- **Optimal Water Clarity**: 80%+ transparency of thin layer

## Troubleshooting

### Common Issues
1. **Poor Detection Accuracy**
   - Ensure proper lighting conditions
   - Check water clarity and background contrast
   - Adjust detection threshold in config.yml

2. **Slow Processing**
   - Reduce input image resolution
   - Enable GPU acceleration
   - Use batch processing for large datasets

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
