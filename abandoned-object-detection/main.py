import cv2
import numpy as np
import time
import math

video_path = 'test_video4.mp4'

ALARM_TIME = 5          # Через сколько секунд считать объект оставленным
STATIC_TIME = 3         # Сколько секунд объект должен быть неподвижным
MIN_AREA = 2000         # Минимальная площадь контура (чтобы отсеять шум)
MAX_DISTANCE = 100      # Максимальное расстояние для сопоставления объектов между кадрами
SIMILARITY_THRESHOLD = 0.7  # Порог схожести объектов (0-1)
BACKGROUND_FRAMES = 30  # Сколько кадров использовать для построения фона

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео")
    exit()

background_frames = []
for i in range(BACKGROUND_FRAMES):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    background_frames.append(gray)

background_gray = np.median(background_frames, axis=0).astype(np.uint8)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

objects = {}
next_object_id = 0

def calculate_similarity(obj1, obj2):
    dist = math.sqrt((obj1['center'][0] - obj2['center'][0])**2 + 
                     (obj1['center'][1] - obj2['center'][1])**2)
    
    area_ratio = min(obj1['area'], obj2['area']) / max(obj1['area'], obj2['area'])
    distance_score = max(0, 1 - dist / MAX_DISTANCE)
    similarity = (distance_score * 0.6 + area_ratio * 0.4)
    
    return similarity, dist

def check_motion_in_roi(frame_gray, prev_frame, bbox, threshold=20):
    x, y, w, h = bbox
    roi_current = frame_gray[y:y+h, x:x+w]
    roi_prev = prev_frame[y:y+h, x:x+w]
    
    if roi_current.shape != roi_prev.shape:
        return True
    
    diff = cv2.absdiff(roi_current, roi_prev)
    motion_pixels = np.sum(diff > threshold)
    return motion_pixels > (w * h * 0.05)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    current_time = time.time()
    
    if 'prev_gray' not in locals():
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
    
    mog_mask = mog.apply(frame)
    mog_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, 
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    frame_diff = cv2.absdiff(background_gray, current_gray)
    _, diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
    diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=3)
    
    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        roi_mog = mog_mask[y:y+h, x:x+w]
        mog_motion = np.sum(roi_mog > 0) > (w * h * 0.1)
        
        local_motion = check_motion_in_roi(current_gray, prev_gray, (x, y, w, h))
        
        current_objects.append({
            'bbox': (x, y, w, h),
            'center': (center_x, center_y),
            'area': area,
            'matched': False,
            'has_motion': mog_motion or local_motion
        })
    
    new_objects = {}
    used_current = set()
    
    for obj_id, obj_data in objects.items():
        best_match = None
        best_similarity = 0
        
        for i, current_obj in enumerate(current_objects):
            if i in used_current:
                continue
                
            similarity, dist = calculate_similarity(obj_data, current_obj)
            
            if similarity > best_similarity and similarity > SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_match = i
        
        if best_match is not None:
            current_obj = current_objects[best_match]
            
            # Логика статичности
            if current_obj['has_motion']:
                # Если есть движение - сбрасываем время статичности
                static_start_time = None
            else:
                # Если нет движения
                if obj_data['static_start_time'] is None:
                    # Начинаем отсчет статичности
                    static_start_time = current_time
                else:
                    # Продолжаем отсчет
                    static_start_time = obj_data['static_start_time']
            
            new_objects[obj_id] = {
                'bbox': current_obj['bbox'],
                'center': current_obj['center'],
                'area': current_obj['area'],
                'start_time': obj_data['start_time'],
                'static_start_time': static_start_time,
                'last_seen': current_time,
                'has_motion': current_obj['has_motion']
            }
            used_current.add(best_match)
    
    for i, current_obj in enumerate(current_objects):
        if i not in used_current:
            # Новый объект - начинаем отсчет статичности если нет движения
            static_start_time = None if current_obj['has_motion'] else current_time
            
            new_objects[next_object_id] = {
                'bbox': current_obj['bbox'],
                'center': current_obj['center'],
                'area': current_obj['area'],
                'start_time': current_time,
                'static_start_time': static_start_time,
                'last_seen': current_time,
                'has_motion': current_obj['has_motion']
            }
            next_object_id += 1
    

    objects = {}
    for obj_id, obj_data in new_objects.items():
        if current_time - obj_data['last_seen'] < 1.0:
            objects[obj_id] = obj_data
    
    
    for obj_id, obj_data in objects.items():
        x, y, w, h = obj_data['bbox']
        time_diff = current_time - obj_data['start_time']
        
        # Сколько секунд объект статичен
        if obj_data['static_start_time'] is not None:
            static_time = current_time - obj_data['static_start_time']
        else:
            static_time = 0
        
        # Определяем цвет и статус
        if static_time >= STATIC_TIME and time_diff >= ALARM_TIME:
            color = (0, 0, 255)  # Красный - тревога
            thickness = 3
            label = f"ALARM! {int(time_diff)}s"
            cv2.putText(frame, "ABANDONED OBJECT!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif static_time >= STATIC_TIME:
            color = (0, 255, 255)  # Желтый - статичный, но еще рано
            thickness = 2
            label = f"{int(time_diff)}s STATIC:{int(static_time)}s"
        elif obj_data['has_motion']:
            color = (255, 0, 0)  # Синий - движется
            thickness = 2
            label = f"{int(time_diff)}s MOVING"
        else:
            color = (0, 255, 0)  # Зеленый - только появился, без движения
            thickness = 2
            label = f"{int(time_diff)}s STATIC:{int(static_time)}s"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.putText(frame, f"Objects: {len(objects)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Abandoned Object Detection', frame)
    
    prev_gray = current_gray.copy()
    
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        background_gray = current_gray.copy()
        objects = {}
        next_object_id = 0
    elif key == ord('c'):
        objects = {}
        next_object_id = 0

cap.release()
cv2.destroyAllWindows()