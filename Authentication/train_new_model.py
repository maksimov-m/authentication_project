import face_recognition
from PIL import Image # Библиотека для работы с изображениями
import os
import cv2

find_face = face_recognition.load_image_file("video/maks1.jpg") # Загружаем изображение нужного человека
face_encoding = face_recognition.face_encodings(find_face)[0] # Кодируем уникальные черты лица, для того чтобы сравнивать с другими
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)

i = 0 # Счётчик общего выполнения
done = 0 # Счётчик совпадений
images = os.listdir('video/')
flag = False
for im in images:
    i += 1 # Увеличиваем счётчик общего выполнения
    unknown_picture = face_recognition.load_image_file(f"video/{im}") # Загружаем скачанное изображение

    print(im)
    image = cv2.imread(f"video/{im}")


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)


    if len(faces) > 0:
        x, y, w, h = faces[0]

        cropped_image = image[y - 80:y + h + 35, x - 20:x + w + 35]

        # Сохранение вырезанного изображения
        #cv2.imwrite('cropped.jpg', cropped_image)

        #image = cv2.imread(f"cropped.jpg")

        rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        #unknown_face_encoding = face_recognition.face_encodings(unknown_picture)  # Кодируем уникальные черты лица
        unknown_face_encoding = face_recognition.face_encodings(rgb)

        pil_image = Image.fromarray(unknown_picture) # Записываем изображение в переменную

        # Проверяем нашла ли нейросеть лицо
        if len(unknown_face_encoding) > 0: # Если нашли лицо
            encoding = unknown_face_encoding[0] # Обращаемся к 0 элементу, чтобы сравнить
            results = face_recognition.compare_faces([face_encoding], encoding) # Сравниваем лица

            if results[0] == True: # Если нашли сходство
                done += 1 # Увеличиваем счётчик общего выполнения
                print(i,"-","Нашли нужного человека !")
                pil_image.save(f"done/{int(done)}.jpg") # Сохраняем фото с найденным человеком
            else: # Если не нашли сходство
                print(i,"-","Не нашли нужного человека!")
        else: # Если не нашли лицо
            print(i,"-","Лицо не найдено!")