import os
os.environ['path'] = os.getcwd() + os.pathsep + os.environ['path']
# print(os.environ['path'])
import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import torch
from src.Net import Net
import torch.nn.functional as F
import heapq
from skimage import transform

LABELS = ('neutral', 'happiness', 'surprise', 'sadness', 'anger',
          'disgust', 'fear', 'contempt', 'unknown', 'NF')
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def resizeTo224(img):
    return transform.resize(img, (224, 224))


def channelTo3(img):
    img = np.expand_dims(img, 2)
    return np.concatenate((img, img, img), axis=2)


class FER(object):
    def __init__(self, trainedEpoch=6):
        self.trainedEpoch = trainedEpoch
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net = Net()
        if not torch.cuda.is_available():
            self.net.load_state_dict(torch.load('models/net' + str(self.trainedEpoch) + '.pkl', map_location='cpu'))
        else:
            self.net.load_state_dict(torch.load('models/net' + str(self.trainedEpoch) + '.pkl'))
        self.net = self.net.to(self.device)
        self.net.eval()
        pass

    def __call__(self, img, onlyOne=True):
        # ResizeTo 224 and Flip axis
        img = resizeTo224(img)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device, dtype=torch.float).unsqueeze_(0)
        predict = F.softmax(self.net(img), dim=1)[0].cpu().detach().numpy()
        top = heapq.nlargest(2, range(len(predict)), predict.take)
        top0 = LABELS[top[0]]
        top1 = LABELS[top[1]]
        score0 = predict[top[0]]
        score1 = predict[top[1]]
        if onlyOne:
            return '{}: {:.2f}'.format(top0, score0)
        else:
            return '{}: {:.2f} {}: {:.2f}'.format(top0, score0, top1, score1)


def main(camera=0):
    # For webcam input:
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    fer = FER(trainedEpoch=14)
    cap = cv2.VideoCapture(camera)
    cap.set(3, 640)
    cap.set(4, 480)
    while cap.isOpened():
        timeStart = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2GRAY)
        image = channelTo3(image)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        imageRaw = image.copy()
        crops = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # single face crop
                x_min, x_max, y_min, y_max = mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                cv2.rectangle(image, (x_min - 1, y_min - 1), (x_max, y_max), (0, 255, 0), 1)
                crops.append(imageRaw[y_min:y_max, x_min:x_max])
        fps = 1. / (time.time() - timeStart)
        cv2.putText(image, f'fps: {round(fps, 1)}', (15, 25), 1, 2, (0, 0, 255), 2)
        if len(crops) > 0:
            crop = crops[0]
            emotion = fer(crop)
            cv2.putText(image, emotion, (300, 25), 1, 2, (0, 0, 255), 2)
            pass
        cv2.imshow('Face', image)
        # print(image.shape)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    face_mesh.close()
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='To assign camera index')
    parser.add_argument('--camera', type=int, default=0, help='camera index')
    arg = parser.parse_args()
    main(arg.camera)