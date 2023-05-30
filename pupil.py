import cv2
import numpy as np

import torch
import torchvision

# Assuming that we are on a CUDA machine, this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define your model architecture here. It should be the same as the one you used during training.
model = torchvision.models.resnet50()

# Load the model weights from a checkpoint file
checkpoint = torch.load('prop_model.pth')

# Load the weights into the model
model.load_state_dict(checkpoint)

# Don't forget to set the model in evaluation mode
model.eval()

# Move the model to GPU if available
model.to(device)

# Load the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the deep learning model
model = load_model('prop_model.h5')

def get_pupil_size(eye):
    # Preprocess the eye image
    eye = cv2.resize(eye, (224, 224))
    eye = eye / 255.0  # normalize to [0,1]
    eye = np.expand_dims(eye, axis=0)  # add batch dimension

    # Use the model to predict pupil coordinates and radius
    [x, y, r] = model.predict(eye)[0]

    # Convert coordinates and radius back to original scale
    x = x * eye.shape[1]
    y = y * eye.shape[0]
    r = r * max(eye.shape[0], eye.shape[1])

    return (x, y, r)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye = roi_color[ey:ey+eh, ex:ex+ew]
            pupil_x, pupil_y, pupil_r = get_pupil_size(eye)
            cv2.circle(roi_color, (int(pupil_x), int(pupil_y)), int(pupil_r), (0, 255, 0), 2)
    cv2.imshow('img', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
