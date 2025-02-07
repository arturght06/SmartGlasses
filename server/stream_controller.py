import cv2


camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Camera not accessible")
else:
    print("Camera works!")
    while True:
        ret, frame = camera.read()
        if ret:
            cv2.imshow('Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error reading frame")
            break
    camera.release()
    cv2.destroyAllWindows()