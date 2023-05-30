# Pupil Size Measurement using PyTorch

This repository contains code for a real-time pupil size measurement system using PyTorch and OpenCV.

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/masonkadem/PupilSizeMeasurement.git
```
2. Navigate to the repository:

```bash
cd PupilSizeMeasurement
```

3. Install the necessary Python libraries:
```bash
pip install -r requirements.txt
```


## Running the Application
To run the application, use the following command:

```bash
python pupil.py
```

The script will start the webcam and begin detecting the pupils in real-time.

## Docker Instructions
You can also run the application in a Docker container.

1. Further DevelopmentBuild the Docker image:

```bash
docker build -t pupil-size-measurement .
```
2. Run the Docker container:
```bash
docker run --rm -p 5000:5000 --name=pupil-size-measurement pupil-size-measurement
```
## Further Development
This application can be extended and improved in several ways, such as improving the accuracy of the pupil size measurement, optimizing the performance for real-time detection, and incorporating more advanced pupil detection methods.
