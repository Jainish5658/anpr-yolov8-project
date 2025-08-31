# Automatic Number Plate Recognition with YOLOv8

A comprehensive automatic number plate recognition (ANPR) system built with Python, YOLOv8, and EasyOCR. This project detects vehicles in video streams, identifies license plates, and extracts the text from them using computer vision and deep learning techniques.

## Features

- **Vehicle Detection**: Uses YOLOv8 pre-trained model to detect cars, trucks, buses, and motorcycles
- **License Plate Detection**: Custom-trained YOLOv8 model specifically for license plate detection
- **Text Recognition**: EasyOCR for optical character recognition of license plate text
- **Multi-Object Tracking**: SORT algorithm for tracking vehicles across video frames
- **Format Validation**: Validates license plate text against standard formats
- **Video Processing**: Processes entire video files and generates annotated output
- **Performance Metrics**: Includes performance analysis and reporting capabilities

## Project Structure

```
.
├── main.py                     # Main processing script
├── util.py                     # Utility functions for OCR and data processing
├── visualize.py                # Video visualization and annotation
├── performance_metrics.py      # Performance analysis tools
├── add_missing_data.py         # Data interpolation utilities
├── requirements.txt            # Python dependencies
├── best.pt                     # Custom trained license plate detection model
├── yolov8n.pt                  # YOLOv8 nano model for vehicle detection
├── sample.mp4                  # Sample input video
├── out.mp4                     # Processed output video
├── test.csv                    # Raw detection results
├── test_interpolated.csv       # Interpolated results
├── sort/                       # SORT tracking algorithm
└── LICENSE                     # GNU AGPL v3 License
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/automatic-number-plate-recognition-python-yolov8.git
   cd automatic-number-plate-recognition-python-yolov8
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SORT module**:
   The SORT module needs to be downloaded from [this repository](https://github.com/abewley/sort):
   ```bash
   git clone https://github.com/abewley/sort.git
   ```

4. **Download models and sample data**:
   - `yolov8n.pt`: YOLOv8 nano model (automatically downloaded by ultralytics)
   - `best.pt`: Custom license plate detection model (included)
   - `sample.mp4`: Download a sample video from [Pexels](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/) or use your own video file

## Usage

### Basic Usage

1. **Process a video**:
   ```bash
   python main.py
   ```
   This will process the `sample.mp4` file and generate detection results in `test.csv`.

2. **Visualize results**:
   ```bash
   python visualize.py
   ```
   This creates an annotated output video `out.mp4` with detected vehicles and license plates.

3. **Analyze performance**:
   ```bash
   python performance_metrics.py
   ```
   Generates performance analysis and metrics.

### Customization

- **Input Video**: Replace `sample.mp4` with your own video file, or modify the path in `main.py`
- **Output Format**: Modify the CSV output format in `util.py`
- **Detection Classes**: Adjust vehicle types in `main.py` (currently: cars, trucks, buses, motorcycles)

## How It Works

1. **Vehicle Detection**: The system uses YOLOv8 to detect vehicles in each video frame
2. **Object Tracking**: SORT algorithm tracks detected vehicles across frames
3. **License Plate Detection**: A custom-trained YOLOv8 model detects license plates within vehicle bounding boxes
4. **Text Extraction**: EasyOCR extracts text from cropped license plate images
5. **Format Validation**: The system validates license plate text against expected formats
6. **Result Output**: Detection results are saved to CSV files for analysis

## Models

### Vehicle Detection Model
A YOLOv8 pre-trained model (`yolov8n.pt`) is used to detect vehicles in the following classes:
- Cars (class_id: 2)
- Motorcycles (class_id: 3)
- Buses (class_id: 5)
- Trucks (class_id: 7)

### License Plate Detection Model
A custom-trained YOLOv8 model (`best.pt`) specifically trained for license plate detection using:
- Dataset: [License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
- Training Guide: [YOLOv8 Custom Dataset Training](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide)

## Dependencies

- **ultralytics**: YOLOv8 implementation
- **opencv-python**: Computer vision operations
- **easyocr**: Optical character recognition
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **filterpy**: Kalman filtering for SORT tracking

## Data Formats

### Output CSV Structure
The system generates CSV files with the following columns:
- `frame_nmr`: Video frame number
- `car_id`: Unique vehicle tracking ID
- `car_bbox`: Vehicle bounding box coordinates [x1, y1, x2, y2]
- `license_plate_bbox`: License plate bounding box coordinates [x1, y1, x2, y2]
- `license_plate_bbox_score`: Confidence score for license plate detection
- `license_number`: Extracted license plate text
- `license_number_score`: Confidence score for text recognition

## Sample Data

The project includes sample data for testing:
- `sample.mp4`: Input video file
- Traffic video can be downloaded from [Pexels](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)

## Performance Metrics

The system includes performance analysis capabilities:
- Detection accuracy metrics
- Processing speed analysis
- Text recognition confidence scores
- Vehicle tracking consistency

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [SORT](https://github.com/abewley/sort) for multi-object tracking
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [Computer Vision Engineer](https://github.com/computervisioneng) for the original tutorial

## Troubleshooting

### Common Issues

1. **Missing SORT module**: Make sure to clone the SORT repository as described in the installation steps
2. **Model files not found**: Ensure `best.pt` and `yolov8n.pt` are in the project directory
3. **GPU issues**: The project is configured to run on CPU by default. For GPU acceleration, modify the `gpu=False` parameter in `util.py`
4. **Video codec issues**: If you encounter video writing problems, try different codecs in `visualize.py`

### Performance Tips

- Use GPU acceleration for faster processing (requires CUDA-compatible GPU)
- Adjust detection confidence thresholds for better accuracy
- Use smaller input videos for faster testing
- Consider using YOLOv8s or YOLOv8m models for better accuracy (at the cost of speed)
