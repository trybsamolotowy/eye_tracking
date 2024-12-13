import cv2
import numpy as np
import argparse
import pyautogui
import json
import time

pyautogui.FAILSAFE = False # błąd gdy kursor znajduje się w rogu ekranu

def load_calibration_data(file_path="calibrate_data.txt"):
    try:
        with open(file_path, "r") as file:
            calibration_data = [eval(line.strip()) for line in file.readlines()]
        print("Dane kalibracyjne załadowane.")
        return calibration_data
    except FileNotFoundError:
        print(f"Plik {file_path} nie został znaleziony.")
        return None

def map_eye_position_to_screen(eye_position, calibration_data, screen_w, screen_h):
    
    closest_point = min(
        calibration_data,
        key=lambda data: np.linalg.norm(
            np.array(eye_position) - np.array(data["eye_position"])
        )
    )
    calibration_point = closest_point["calibration_point"]

    screen_x = calibration_point[0]
    screen_y = calibration_point[1]

    screen_x = max(0, min(screen_w - 1, screen_x))
    screen_y = max(0, min(screen_h - 1, screen_y))

    return screen_x, screen_y

last_screen_x, last_screen_y = None, None
alpha = 0.2  # Współczynnik wygładzania


def position_cursor(cap):
    global last_screen_x, last_screen_y
    screen_w, screen_h = pyautogui.size()
    calibration_data = load_calibration_data()

    if calibration_data is None:
        print("Brak danych kalibracyjnych. Uruchom najpierw kalibrację.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, binary_frame = cv2.threshold(gray_frame, 45, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)

            if moments["m00"] != 0:
                cx_eye = int(moments["m10"] / moments["m00"])
                cy_eye = int(moments["m01"] / moments["m00"])

                # Mapowanie na ekran
                screen_x, screen_y = map_eye_position_to_screen(
                    (cx_eye, cy_eye), calibration_data, screen_w, screen_h
                )

                # Wygładzanie pozycji kursora
                if last_screen_x is not None and last_screen_y is not None:
                    screen_x = int(alpha * screen_x + (1 - alpha) * last_screen_x)
                    screen_y = int(alpha * screen_y + (1 - alpha) * last_screen_y)

                last_screen_x, last_screen_y = screen_x, screen_y

                pyautogui.moveTo(screen_x, screen_y)

        cv2.imshow("Pozycjonowanie kursora", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def stream_video(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        _, binary_frame = cv2.threshold(gray_frame, 45, 255, cv2.THRESH_BINARY_INV)

       
        cv2.imshow("Oryginalne wideo", frame)
        cv2.imshow("Skala szarości", gray_frame)
        cv2.imshow("Binaryzacja", binary_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def capture_contours(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, binary_frame = cv2.threshold(gray_frame, 45, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            step = max(1, len(largest_contour) // 5)  #
            points = largest_contour[::step]

            binary_frame_with_contours = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)

            cv2.drawContours(binary_frame_with_contours, [largest_contour], -1, (0, 255, 0), 2)

            for i, point in enumerate(points):
                if i >= 20:  
                    break
                x, y = point[0]
                cv2.putText(binary_frame_with_contours, str(i + 1), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow("Kontury i numeracja", binary_frame_with_contours)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def calibrate(cap):
    screen_w, screen_h = pyautogui.size()
    calibration_points = [
        (screen_w // 2, screen_h // 2),  # Środek
        (0, 0),  
        (screen_w - 1, 0),  
        (screen_w - 1, screen_h - 1),  
        (0, screen_h - 1),  
        (screen_w // 2, 0),  
        (screen_w - 1, screen_h // 2),  
        (screen_w // 2, screen_h - 1),  
        (0, screen_h // 2),
    ]

    cv2.namedWindow("Kalibracja", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Kalibracja", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calibration_data = []

    for i, (cx, cy) in enumerate(calibration_points):
        calibration_screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(calibration_screen, (cx, cy), 20, (0, 255, 0), -1)
        cv2.imshow("Kalibracja", calibration_screen)

        start_time = time.time()

        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, binary_frame = cv2.threshold(gray_frame, 45, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                moments = cv2.moments(largest_contour)

                if moments["m00"] != 0:
                    cx_eye = int(moments["m10"] / moments["m00"])
                    cy_eye = int(moments["m01"] / moments["m00"])

                    calibration_data.append({
                        "calibration_point": (cx, cy),
                        "eye_position": (cx_eye, cy_eye),
                        "time": time.time()
                    })

        cv2.waitKey(500)

    with open("calibrate_data.txt", "w") as file:
        for data in calibration_data:
            file.write(f"{data}\n")

    cv2.destroyAllWindows()
    
# flagi
def main():
    parser = argparse.ArgumentParser(description="Sterowanie kursorem za pomocą źrenicy.")
    parser.add_argument("--calibracja", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--capture_contours", action="store_true")
    parser.add_argument("--pozycjonowanie_kursora", action="store_true")

    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie można otworzyć kamery.")
        exit()

    if args.calibracja:
        calibrate(cap)
    elif args.stream:
        stream_video(cap)
    elif args.capture_contours:
        capture_contours(cap)
    elif args.pozycjonowanie_kursora:
        position_cursor(cap)
    else:
        print("Podaj jedną z flag: --calibracja, --stream, --capture_contours, --pozycjonowanie_kursora")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
