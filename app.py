import time
import numpy as np
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp
from flask_socketio import SocketIO
import pyautogui
from pose_module import *


app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index2.html')

def gen():
    previous_time = 0
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detected our pose
    my_pose = mp.solutions.pose
    pose = my_pose.Pose()

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # converting image to RGB from BGR cuz mediapipe only work on RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        # print(result.pose_landmarks)
        if result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks, my_pose.POSE_CONNECTIONS)

        # checking video frame rate
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Writing FrameRate on video
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        #cv2.imshow("Pose detection", img)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


def face_detect():

    mp_facedetector = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

        while cap.isOpened():

            success, image = cap.read()

            start = time.time()

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            results = face_detection.process(image)

            # Convert the image color back so it can be displayed
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.detections:
                for id, detection in enumerate(results.detections):
                    mp_draw.draw_detection(image, detection)
                    # print(id, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = image.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    cv2.putText(image, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            # print("FPS: ", fps)

            cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # cv2.imshow('Face Detection', image)
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(5) & 0xFF == 27:
                break


def hand_detect():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            screenWidth, screenHeight = pyautogui.size()
            # Check if either hand is detected.
            action = None
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    action = detect_gestures(hand_landmarks)
                    try:
                        x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * screenWidth
                        y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * screenHeight
                        z = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].z

                        pyautogui.moveTo(x, y)

                    except Exception as e:
                        print(e)

                    try:
                        if (action == 'refresh'):
                            print('REFRESH')
                            pyautogui.hotkey('command', 'r')


                    except Exception as e:
                        print(e)

                    try:
                        if (action == 'zoomIn'):
                            pyautogui.hotkey('command', '+')
                    except Exception as e:
                        print(e)

                    try:
                        if (action == 'zoomOut'):
                            pyautogui.hotkey('command', '-')
                    except Exception as e:
                        print(e)

                    try:
                        if (action == 'scrollUp'):
                            pyautogui.scroll(1)

                    except Exception as e:
                        print(e)

                    try:
                        if (action == 'scrollDn'):
                            pyautogui.scroll(-1)

                    except Exception as e:
                        print(e)

                    # try:
                    #     if (action == 'select'):
                    #         pyautogui.click()
                    # except Exception as e:
                    #     print(e)


            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # cv2.imshow('MediaPipe Hands', image)

            cv2.putText(image, f'{action}', (25, 65), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 2)
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(5) & 0xFF == 27:
                break



def holistic_detect():
    t0 = time.time()

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    lav = (187, 160, 178)
    pink = (172, 18, 143)
    red = (58, 45, 240)
    white = (0, 0, 0)
    black = (0, 0, 0)

    pose_samples_folder = 'poses_csvs_out'

    pose_embedder = FullBodyPoseEmbedder()

    pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    cap = cv2.VideoCapture(0)

    t1 = time.time()
    # print(f"Took {t1 - t0} seconds for setup")

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            t0 = time.time()
            success, image = cap.read()

            start = time.time()

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            image.flags.writeable = False

            results = holistic.process(image)


            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            t1 = time.time()
            # print(f"Took {t1 - t0} seconds to get frame and classifiy using holistic model")

            lmColor = pink
            connColor = lav
            rad = 9
            thick = 4

            mp_drawing.draw_landmarks(image,
                                      results.left_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=(rad // 2)),
                                      mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=(rad // 2)),
                                      mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),
                                      mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))

            t0 = time.time()
            output_frame = image.copy()
            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array(
                    [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in
                     pose_landmarks.landmark],
                    dtype=np.float32)
                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                t1 = time.time()
                # print(f"Took {t1 - t0} seconds to format the pose landmarks")

                pose_classification = pose_classifier(pose_landmarks)
                print('Pose classification: {}'.format(pose_classification))
                action = max(pose_classification, key=pose_classification.get)

                if action == 'select':
                    # socketio.emit('move_box', {'direction': 'up'})
                    socketio.emit('select_image')
                elif action == 'refresh':
                    # socketio.emit('move_box', {'direction': 'down'})
                    socketio.emit('refresh_image')
                elif action == 'scrollUp':
                    socketio.emit('scroll_image', {'direction': 'up'})
                elif action == 'scrollDown':
                    socketio.emit('scroll_image', {'direction': 'down'})
                elif action == 'scrollLeft':
                    socketio.emit('scroll_image', {'direction': 'left'})
                elif action == 'scrollRight':
                    socketio.emit('scroll_image', {'direction': 'right'})
                elif action == 'zoomIn':
                    socketio.emit('zoom_image', {'direction': 'in'})
                elif action == 'zoomOut':
                    socketio.emit('zoom_image', {'direction': 'out'})
            else:
                action = None
                # t1 = time.time()
                # print(f"Took {t1 - t0} seconds to classify the pose")

            screenWidth, screenHeight = pyautogui.size()

            textColor = white

            try:
                x = results.left_hand_landmarks.landmark[8].x * screenWidth
                y = results.left_hand_landmarks.landmark[8].y * screenHeight
                z = results.left_hand_landmarks.landmark[8].z

                pyautogui.moveTo(x, y)

            except Exception as e:
                print(e)

            try:
                if (action == 'refresh') and (pose_classification[action] == 10):
                    print('REFRESH')
                    pyautogui.hotkey('command', 'r')


            except Exception as e:
                print(e)

            try:
                if (action == 'scrollUp') and (pose_classification[action] == 10):
                    pyautogui.scroll(5)

            except Exception as e:
                print(e)

            try:
                if (action == 'scrollDn') and (pose_classification[action] == 10):
                    pyautogui.scroll(-5)

            except Exception as e:
                print(e)

            try:
                x_thumb = results.right_hand_landmarks.landmark[4].x * screenWidth
                y_thumb = results.right_hand_landmarks.landmark[4].y * screenHeight

                x_pinky = results.right_hand_landmarks.landmark[20].x * screenWidth
                y_pinky = results.right_hand_landmarks.landmark[20].y * screenHeight

                distance = ((((x_thumb - x_pinky) ** 2) + ((y_thumb - y_pinky) ** 2)) ** 0.5)

                if distance < 20:

                    pyautogui.mouseDown()
                    cv2.putText(image, 'select', (25, 140), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2)

                elif distance > 20:
                    pyautogui.mouseUp()
                    cv2.putText(image, 'unSelect', (25, 140), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2)


            except Exception as e:
                print(e)


            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            print("FPS: ", fps)

            action_stack = fps

            capWidth = cap.get(3)
            capHeight = cap.get(4)

            scale = 0.5
            # newCapWidth = int(screenWidth * (scale))
            # newCapHeight = int(newCapWidth * capHeight / capWidth)

            # capX = screenWidth - newCapWidth - 5
            # capY = screenHeight - newCapHeight - 33

            cv2.putText(image, f'FPS: {int(fps)}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(image, f'{action}', (25, 65), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2)
            # cv2.imshow('Choreographic Interface', cv2.resize(image, (newCapWidth, newCapHeight)))
            # cv2.setWindowProperty('Choreographic Interface', cv2.WND_PROP_TOPMOST, 1)
            # cv2.moveWindow('Choreographic Interface', capX, capY)

            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(5) & 0xFF == 27:
                break


def detect_gestures(hand_landmarks, previous_landmarks=None):
    # Define indices for fingertips and palm base in MediaPipe hand landmarks.
    TIP_IDS = [mp.solutions.hands.HandLandmark.THUMB_TIP,
               mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
               mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
               mp.solutions.hands.HandLandmark.PINKY_TIP]

    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_bottom = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_CMC]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_bottom = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_bottom = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    pinky_bottom = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]
    print(thumb_bottom.y-thumb_tip.y, index_bottom.y-index_tip.y, ring_bottom.y-ring_tip.y, pinky_bottom.y-pinky_tip.y)
    if thumb_bottom.y-thumb_tip.y > 0.4 and index_bottom.y-index_tip.y < 0.3 and \
            ring_bottom.y-ring_tip.y < 0.3 and pinky_bottom.y-pinky_tip.y < 0.3:
        action = 'scrollUp'
    elif thumb_tip.y-thumb_bottom.y > 0.4 and index_bottom.y-index_tip.y < 0.3 and \
            ring_bottom.y-ring_tip.y < 0.3 and pinky_bottom.y-pinky_tip.y < 0.3:
        action = 'scrollDn'
    elif index_bottom.y - index_tip.y > 0.4 and thumb_bottom.y - thumb_tip.y < 0.4 and \
            ring_bottom.y - ring_tip.y < 0.3 and pinky_bottom.y - pinky_tip.y < 0.3:
        action = 'zoomIn'
    elif pinky_bottom.y - pinky_tip.y > 0.3 and index_bottom.y - index_tip.y < 0.3 and \
            ring_bottom.y - ring_tip.y < 0.3 and thumb_bottom.y - thumb_tip.y < 0.4:
        action = 'zoomOut'
    elif thumb_bottom.y - thumb_tip.y <0.3 and index_bottom.y - index_tip.y <0 and \
            ring_bottom.y - ring_tip.y <0 and pinky_bottom.y - pinky_tip.y <0:
        action = 'select'
    elif thumb_bottom.y - thumb_tip.y > 0.2 and index_bottom.y - index_tip.y > 0.2 and \
            ring_bottom.y - ring_tip.y > 0.2 and pinky_bottom.y - pinky_tip.y > 0.2:
        action = 'refresh'
    else:
        action = None
    # Return a dictionary with the detected gestures.

    return action


# This variable should be persistent across calls to scroll_hand_movement, e.g., as part of a class or a global variable.
previous_wrist_position = None


def scroll_hand_movement(hand_landmarks):
    global previous_wrist_position
    # Extract the wrist landmark from the hand landmarks.
    wrist_landmark = hand_landmarks.landmark[0]

    # Initialize scroll direction.
    scroll_direction = None

    # Check if we have a previous wrist position to compare with.
    if previous_wrist_position:
        # Calculate the change in the y-coordinate.
        delta_y = wrist_landmark.y - previous_wrist_position.y

        # Determine the scroll direction based on the change in y-coordinate.
        if abs(delta_y) > 5:  # Replace 'some_threshold' with a value that works for your application.
            if delta_y > 0:
                scroll_direction = 'scrollDown'
            elif delta_y < 0:
                scroll_direction = 'scrollUp'

    # Update the previous wrist position for the next call.
    previous_wrist_position = wrist_landmark

    return scroll_direction


def simple_design():
    t0 = time.time()

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    lav = (187, 160, 178)
    pink = (172, 18, 143)
    red = (58, 45, 240)
    white = (0, 0, 0)
    black = (0, 0, 0)

    pose_samples_folder = 'poses_csvs_out'

    pose_embedder = FullBodyPoseEmbedder()

    pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    cap = cv2.VideoCapture(0)

    t1 = time.time()
    # print(f"Took {t1 - t0} seconds for setup")

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            t0 = time.time()
            success, image = cap.read()

            start = time.time()

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            image.flags.writeable = False

            results = holistic.process(image)

            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            t1 = time.time()
            # print(f"Took {t1 - t0} seconds to get frame and classifiy using holistic model")

            lmColor = pink
            connColor = lav
            rad = 9
            thick = 4

            mp_drawing.draw_landmarks(image,
                                      results.left_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=(rad // 2)),
                                      mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=(rad // 2)),
                                      mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=lmColor, thickness=thick, circle_radius=rad),
                                      mp_drawing.DrawingSpec(color=connColor, thickness=thick, circle_radius=rad))

            t0 = time.time()
            output_frame = image.copy()

            screenWidth, screenHeight = pyautogui.size()

            action = None
            try:
                y_thumb = results.right_hand_landmarks.landmark[4].y * screenHeight
                y_thumb_middle = results.right_hand_landmarks.landmark[3].y * screenHeight
                print('y_thumb', y_thumb)
                print('y_thumb_middle', y_thumb_middle)
                if y_thumb > y_thumb_middle:
                    action = 'refresh'
            except Exception as e:
                print(e)
                pass

            try:
                y_2nd_finger = results.right_hand_landmarks.landmark[8].y * screenHeight
                y_2nd_finger_middle = results.right_hand_landmarks.landmark[7].y * screenHeight
                if y_2nd_finger > y_2nd_finger_middle:
                    action = 'zoomIn'
            except:
                pass
            try:
                y_4th_finger = results.right_hand_landmarks.landmark[16].y * screenHeight
                y_4th_finger_middle = results.right_hand_landmarks.landmark[15].y * screenHeight
                if y_4th_finger > y_4th_finger_middle:
                    action = 'zoomOut'
            except:
                pass
            try:
                y_pinky = results.right_hand_landmarks.landmark[20].y * screenHeight
                y_pinky_middle = results.right_hand_landmarks.landmark[19].y * screenHeight
                if y_pinky > y_pinky_middle:
                    action = 'close'
            except:
                pass

            try:
                scroll_direction = scroll_hand_movement(results.right_hand_landmarks)
                if scroll_direction == 'scrollUp':
                    action = 'scrollUp'
                elif scroll_direction == 'scrollDown':
                    action = 'scrollDown'
            except:
                pass


            textColor = white

            try:
                x = results.left_hand_landmarks.landmark[8].x * screenWidth
                y = results.left_hand_landmarks.landmark[8].y * screenHeight
                z = results.left_hand_landmarks.landmark[8].z

                pyautogui.moveTo(x, y)

            except Exception as e:
                print(e)

            try:
                if (action == 'refresh'):
                    print('REFRESH')
                    pyautogui.hotkey('command', 'r')


            except Exception as e:
                print(e)

            try:
                if (action == 'scrollUp'):
                    pyautogui.scroll(5)

            except Exception as e:
                print(e)

            try:
                if (action == 'scrollDown'):
                    pyautogui.scroll(-5)

            except Exception as e:
                print(e)

            try:
                x_thumb = results.right_hand_landmarks.landmark[4].x * screenWidth
                y_thumb = results.right_hand_landmarks.landmark[4].y * screenHeight

                x_2 = results.right_hand_landmarks.landmark[8].x * screenWidth
                y_2 = results.right_hand_landmarks.landmark[8].y * screenHeight

                distance = ((((x_thumb - x_2) ** 2) + ((y_thumb - y_2) ** 2)) ** 0.5)

                if distance < 20:

                    pyautogui.mouseDown()
                    cv2.putText(image, 'select', (25, 140), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2)

                elif distance > 20:
                    pyautogui.mouseUp()
                    cv2.putText(image, 'unSelect', (25, 140), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2)


            except Exception as e:
                print(e)

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            print("FPS: ", fps)


            cv2.putText(image, f'FPS: {int(fps)}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(image, f'{action}', (25, 65), cv2.FONT_HERSHEY_TRIPLEX, 1.5, black, 2)

            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(5) & 0xFF == 27:
                break


@app.route('/video_feed')
def video_feed():
    return Response(hand_detect(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)







