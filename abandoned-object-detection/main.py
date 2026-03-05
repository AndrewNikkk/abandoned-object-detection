import cv2
import numpy as np
import time

video_path = 'test_video4.mp4'
ALARM_TIME = 5
MIN_AREA = 2000

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео")
    exit()

ret, background_frame = cap.read()
if not ret:
    exit()

background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

objects_time = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
    
    frame_diff = cv2.absdiff(background_gray, current_gray)
    _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
    diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=3)
    
    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_objects = {}
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        zone_key = (center_x // 50, center_y // 50)
        
        current_objects[zone_key] = {
            'bbox': (x, y, w, h),
            'center': (center_x, center_y),
            'area': area
        }
        
        if zone_key not in objects_time:
            objects_time[zone_key] = current_time
        else:
            time_diff = current_time - objects_time[zone_key]
            
            if time_diff >= ALARM_TIME:
                color = (0, 0, 255)
                thickness = 3
                label = f"ALARM! {int(time_diff)}s"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, "ABANDONED OBJECT!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                color = (0, 255, 0)
                thickness = 2
                label = f"{int(time_diff)}s"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    objects_time = {k: v for k, v in objects_time.items() if k in current_objects}
    
    cv2.imshow('Abandoned Object Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        background_gray = current_gray.copy()
        objects_time = {}
    elif key == ord('c'):
        objects_time = {}

cap.release()
cv2.destroyAllWindows()