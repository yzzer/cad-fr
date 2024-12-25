import cv2


def select_available_camera() -> cv2.VideoCapture:
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        cap.release()
    raise ValueError("No available camera found")

def get_web_camera(url) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        return cap
    cap.release()
    raise ValueError("No available camera found")
    
if __name__ == '__main__':
    cap = get_web_camera("http://10.29.6.99:4747")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()