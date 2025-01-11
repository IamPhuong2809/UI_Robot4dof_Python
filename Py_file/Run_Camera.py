import cv2

# Tọa độ crop
crop_x1, crop_y1 = 99, 93
crop_x2, crop_y2 = 609, 284

# Hàm xử lý sự kiện chuột
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Kiểm tra sự kiện nhấn chuột trái
        print(f"Tọa độ pixel: X = {x}, Y = {y}")
        
        # Lấy màu của pixel tại tọa độ (x, y)
        if crop_x1 <= x < crop_x2 and crop_y1 <= y < crop_y2:  # Kiểm tra nếu tọa độ nằm trong vùng crop
            pixel_color = frame[y, x]  # OpenCV sử dụng (y, x) thay vì (x, y)
            b, g, r = pixel_color
            print(f"Màu pixel tại ({x}, {y}): R = {r}, G = {g}, B = {b}")
        else:
            print("Tọa độ nằm ngoài vùng crop!")

# Mở camera (0 là ID của camera mặc định, thay đổi nếu cần)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

# Gắn sự kiện chuột
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", on_mouse_click)

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc dữ liệu từ camera!")
        break
    
    # Cắt khung hình theo vùng crop
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Hiển thị khung hình đã cắt
    cv2.imshow("Camera", cropped_frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
