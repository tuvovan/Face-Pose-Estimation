## need opencv, numpy, imutils and dlib installed
## tuvovan


import cv2
import numpy as np 
from imutils import face_utils
import dlib

# get 2D points
def get_2D_points(subject):
    shape = predictor(gray, subject)
    shape = face_utils.shape_to_np(shape) 
    left_eye_left = shape[36]
    right_eye_right = shape[45]
    nose_top = shape[33]
    chin = shape[8]
    left_mouth_conner = shape[48]
    right_mouth_conner = shape[54]

    return np.array([nose_top, chin, left_eye_left, right_eye_right, left_mouth_conner, right_mouth_conner], dtype='double')

## 3D points
model_points = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ]
)

## Get pose
def get_pose(model_points, image_points, frame):
    size = frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0 , 1]
        ], dtype = 'double'
    )

    dist_coeffs = np.zeros((4,1))


    (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0 , 255), -1)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(frame, p1, p2, (255, 0 , 0), 2)


cap = cv2.VideoCapture(0)
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600, 600))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    for subject in subjects:
        image_points = get_2D_points(subject)
        get_pose(model_points, image_points, frame)
    frame = cv2.flip(frame, 1)
    cv2.imshow('out', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()