import cv2
import numpy as np
import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

model_points_center = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-200.0, 170.0, -135.0),  # Left eye center
    (200.0, 170.0, -135.0),  # Right eye center
])

size = (480, 640)
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion


def draw3DAxis(img, origin=(50, 60)):
    cv2.arrowedLine(img, origin, (origin[0] + 55, origin[1]), (0, 0, 255), thickness=3)
    cv2.putText(img, 'X', (origin[0] + 40, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
    cv2.arrowedLine(img, origin, (origin[0], origin[1] - 50), (0, 255, 0), thickness=3)
    cv2.putText(img, 'Y', (origin[0] + 10, origin[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
    cv2.arrowedLine(img, origin, (origin[0] - 35, origin[1] + 35), (255, 0, 0), thickness=3)
    cv2.putText(img, 'Z', (origin[0] - 25, origin[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=2)
    return


def draw3Dline(img, image_point, model_point, rotation_vector, translation_vector):
    (end_point2D, jacobian) = cv2.projectPoints(
        np.array([(model_point[0], model_point[1], 1000.0)]),
        rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p1 = (int(image_point[0]), int(image_point[1]))
    p2 = (int(end_point2D[0][0][0]), int(end_point2D[0][0][1]))
    cv2.line(img, p1, p2, (0, 0, 255), 2)


def main(camera=0):
    cap = cv2.VideoCapture(camera)
    cap.set(3, 640)
    cap.set(4, 480)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        draw3DAxis(image)
        results = face_mesh.process(image)
        kps = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_min, x_max, y_min, y_max, kps = mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        if kps is not None:
            image_points = kps[0:6]
            for p in image_points:
                cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            flags = 8  # most stable pnp solver
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs,
                                                                          flags=flags)
            # draw line from left eye center
            model_point = model_points_center[2]
            image_point = ((kps[2][0] + kps[6][0])/2., (kps[2][1] + kps[6][1])/2.)
            draw3Dline(image, image_point, model_point, rotation_vector, translation_vector)
            # draw line from right eye center
            model_point = model_points_center[3]
            image_point = ((kps[3][0] + kps[7][0])/2., (kps[3][1] + kps[7][1])/2.)
            draw3Dline(image, image_point, model_point, rotation_vector, translation_vector)

        cv2.imshow('Face', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    face_mesh.close()
    cap.release()


if __name__ == '__main__':
    main()
