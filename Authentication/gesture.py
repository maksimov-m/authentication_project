import cv2
import mediapipe as mp
from mediapipe import ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Инициализация объекта MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Загрузка видеопотока (0 - камера по умолчанию)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Захват кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Передача кадра в модель
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:, :, ::-1]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = recognizer.recognize(image)
    if len(recognition_result.gestures) > 0:
        #print(recognition_result.gestures)
        top_gesture = recognition_result.gestures[0][0]
        results = hands.process(frame_rgb)
        cv2.putText(frame, top_gesture.category_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Проверка наличия руки



    # Отображение кадра
    cv2.imshow('Hand Gesture Recognition', frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()