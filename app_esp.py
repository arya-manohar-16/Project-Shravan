#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import socket
import time
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  type=int, default=0)
    parser.add_argument("--width",   type=int, default=960)
    parser.add_argument("--height",  type=int, default=540)

    # ── ESP8266 UDP target ───────────────────────────────────────────────────
    # After connecting your PC to the "ESP8266_CAR" Wi-Fi hotspot,
    # the ESP8266 is always reachable at this fixed IP and port.
    parser.add_argument("--esp_ip",   type=str, default='192.168.4.1')
    parser.add_argument("--esp_port", type=int, default=4210)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    return parser.parse_args()


# ── Gesture ID → Command character ──────────────────────────────────────────
GESTURE_COMMAND_MAP = {
    0: 'F',   # Forward
    1: 'S',   # Stop
    2: 'B',   # Backward
}
# ────────────────────────────────────────────────────────────────────────────


class UDPSender:
    """
    Wraps a UDP socket so the rest of the code looks identical
    to the old serial version — just call send(char).
    UDP is connectionless so there is nothing to 'connect' to;
    we simply send datagrams to the ESP8266's IP:port.
    """
    def __init__(self, ip, port):
        self.target = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # No bind needed on the sender side
        print(f"[UDP] Sender ready → {ip}:{port}")
        print(f"[UDP] Make sure your PC is connected to the 'ESP8266_CAR' Wi-Fi first!")

    def send(self, command_char):
        try:
            data = (command_char + '\n').encode()
            self.sock.sendto(data, self.target)
        except Exception as e:
            print(f"[UDP] Send error: {e}")

    def close(self):
        self.sock.close()


def main():
    args = get_args()

    # ── UDP sender ───────────────────────────────────────────────────────────
    udp = UDPSender(args.esp_ip, args.esp_port)

    use_static_image_mode    = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence  = args.min_tracking_confidence
    use_brect = True

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier      = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history         = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode         = 0
    last_command = None     # Only send when gesture changes

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:       # ESC → quit
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect         = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list      = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                logging_csv(number, mode,
                            pre_processed_landmark_list,
                            pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # ── UDP command dispatch ─────────────────────────────────────
                command_char = GESTURE_COMMAND_MAP.get(hand_sign_id)
                if command_char and command_char != last_command:
                    udp.send(command_char)
                    last_command = command_char
                    print(f"[GESTURE] ID={hand_sign_id}  →  '{command_char}'")
                # ────────────────────────────────────────────────────────────

                if hand_sign_id == 0:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image, brect, handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            # ── Safety stop when hand disappears ────────────────────────────
            if last_command != 'S':
                udp.send('S')
                last_command = 'S'
                print("[SAFETY] No hand detected → Stop")
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    udp.send('S')   # Safety stop on exit
    udp.close()
    cap.release()
    cv.destroyAllWindows()


# ── All original helper functions (unchanged) ────────────────────────────────

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == 110: mode = 0
    if key == 107: mode = 1
    if key == 104: mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat))
    return [v / max_val for v in flat]

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for i, point in enumerate(temp):
        if i == 0:
            base_x, base_y = point[0], point[1]
        temp[i][0] = (temp[i][0] - base_x) / image_width
        temp[i][1] = (temp[i][1] - base_y) / image_height
    return list(itertools.chain.from_iterable(temp))

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 1 and 0 <= number <= 9:
        with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])
    if mode == 2 and 0 <= number <= 9:
        with open('model/point_history_classifier/point_history.csv', 'a', newline="") as f:
            csv.writer(f).writerow([number, *point_history_list])

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        connections = [
            (2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
            (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
            (0,1),(1,2),(2,5),(5,9),(9,13),(13,17),(17,0)
        ]
        for s, e in connections:
            cv.line(image, tuple(landmark_point[s]), tuple(landmark_point[e]), (0,0,0), 6)
            cv.line(image, tuple(landmark_point[s]), tuple(landmark_point[e]), (255,255,255), 2)
        for i, lm in enumerate(landmark_point):
            r = 8 if i in [4,8,12,16,20] else 5
            cv.circle(image, (lm[0], lm[1]), r, (255,255,255), -1)
            cv.circle(image, (lm[0], lm[1]), r, (0,0,0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0,0,0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1]-22), (0,0,0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text += ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0]+5, brect[1]-4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    if finger_gesture_text:
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10,60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10,60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for i, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1+int(i/2), (152,251,152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:"+str(fps), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:"+str(fps), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:"+mode_string[mode-1], (10,90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:"+str(number), (10,110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()