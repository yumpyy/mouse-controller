import cv2
import mediapipe as mp 
import pyautogui as pg

def main():
    camera = cv2.VideoCapture(0)
    screenRes = pg.size()

    solutions = mp.solutions

    handSolutions = solutions.hands 
    drawingStyles = solutions.drawing_styles
    drawing = solutions.drawing_utils

    hands = handSolutions.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    try:
        while camera.isOpened():
            success, vid = camera.read()

            if not success:
                continue

            vid = cv2.flip(vid, 1)
            vid.flags.writeable = False

            vid = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
            results = hands.process(vid)

            vid.flags.writeable = True
            vid = cv2.cvtColor(vid, cv2.COLOR_RGB2BGR)
            
            if (results.multi_hand_landmarks):
                for handLandmarks in results.multi_hand_landmarks:
                    # pg.locateCenterOnScreen()
                    cursorHand = handLandmarks.landmark[8]
                    # print(f"[x] {cursorHand.x} | [y] {cursorHand.y}")

                    cursorPosX = screenRes.width * cursorHand.x 
                    cursorPosY = screenRes.height * cursorHand.y 

                    print(f"[x] {cursorPosX}\t\t[y] {cursorPosY}")
                        
                    pg.mouseDown(button="left")
                    pg.moveTo(cursorPosX, cursorPosY)
                    # pg.mouseUp(button='left', cursorPosX, cursorPosY)

                    drawing.draw_landmarks(vid, handLandmarks, handSolutions.HAND_CONNECTIONS, drawingStyles.get_default_hand_landmarks_style(), drawingStyles.get_default_hand_connections_style())
            
            cv2.imshow("Hands", vid)
            
            if (cv2.waitKey(5) & 0xFF == 27):
                break
    except KeyboardInterrupt:
        print("Exiting")

    camera.release()

if __name__ == "__main__":
    main()
