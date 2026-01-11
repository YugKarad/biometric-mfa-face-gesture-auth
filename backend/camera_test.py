import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera not accessible")
    exit()

print(" Camera accessed successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Phase 1 - Camera Test (Press Q to Exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
