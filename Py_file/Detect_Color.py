import cv2
import numpy as np

# Đọc ảnh template
template = cv2.imread('picture/Nut.jpg', cv2.IMREAD_GRAYSCALE)
h, w = template.shape[:2]  # Lấy kích thước template
posx = np.zeros(20)
posy = np.zeros(20)

# Mở camera
cap = cv2.VideoCapture(0)  # `1` là camera phụ, thay đổi nếu cần

if not cap.isOpened():
    print("Không mở được camera")
    exit()

# Tỷ lệ pixel sang mm (12 pixel = 10mm)
pixel_to_mm = 10 / 12  # 0.8333 mm per pixel

# Tọa độ gốc
origin_x, origin_y = 248, 52


def non_max_suppression(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(scores)
    selected_boxes = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        selected_boxes.append(boxes[i])

        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return selected_boxes

# Các tọa độ để crop
crop_x1, crop_y1 = 99, 93
crop_x2, crop_y2 = 609, 284

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình từ camera")
        break

    # Crop khung hình theo tọa độ đã cho
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Chuyển đổi khung hình từ camera sang grayscale
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # Áp dụng template matching
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

    # Ngưỡng phát hiện
    threshold = 0.7
    locations = np.where(result >= threshold)

    boxes = []
    scores = []
    for pt in zip(*locations[::-1]):
        top_left = pt
        bottom_right = (top_left[0] + w, top_left[1] + h)
        boxes.append([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
        scores.append(result[pt[1], pt[0]])

    selected_boxes = non_max_suppression(boxes, scores, 0.3)
    mm_coordinates = []
    for i, box in enumerate(selected_boxes):
        top_left = (box[0], box[1])
        bottom_right = (box[2], box[3])

        # Tính tọa độ tâm
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2

        # Tính tọa độ từ gốc (origin) và chuyển sang mm
        offset_x = center_x - origin_x
        offset_y = center_y - origin_y
        mm_x = offset_x * pixel_to_mm
        mm_y = offset_y * pixel_to_mm

        mm_coordinates.append((round(mm_x, 3), round(mm_y, 3)))

        # Cắt vùng từ ảnh gốc
        cropped = cropped_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Chuyển đổi vùng cắt sang không gian màu HSV
        hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        average_hue = np.mean(hsv_cropped[:, :, 0])
        average_saturation = np.mean(hsv_cropped[:, :, 1])
        if average_hue < 108 and average_saturation < 25:
            print(average_hue)
            print(average_saturation)
            color = 'Gray'
        elif average_hue < 112 and average_saturation > 40:
            print(average_hue)
            print(average_saturation)
            color = 'Blue'
        elif average_hue > 110 and average_saturation > 24 and average_saturation < 50:
            print(average_hue)
            print(average_saturation)            
            color = 'Red'
        else:
            print(average_hue)
            print(average_saturation)
            color = 'Other'

        # Hiển thị tâm và thông tin màu sắc trên khung hình
        cv2.circle(cropped_frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(cropped_frame, f"{color} ,{mm_x:.3f}, {mm_y:.3f}",
                    (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Hiển thị khung hình đã cắt
    cv2.imshow('Cropped Frame', cropped_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
