import cv2
import numpy as np
from keras.models import model_from_json

# klasördeki duygu sırasına göre bir duygu sözlüğü oluşturuldu

emotion_classes = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and loaded model

json_file = open('model/emotion_model.json','r')
loaded_model = json_file.read()
json_file.close()

#created model
emotion_model = model_from_json(loaded_model)

# load weights into new model
# duygu modelindeki tüm öğrenmeler h5 dosyasında
emotion_model.load_weights("model/emotion_model.h5")
print("loaded model")

# loaded video for model testing

#cap = cv2.VideoCapture(0) : kamera için
cap = cv2.VideoCapture("videos/duygular5.mp4")

while True:

    # find haar cascade to draw bouinding box around face

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280,720)) #frame aldık ve yeniden boyutlandırdık

    if not ret:
        break
    
    #yüz ifadesini bulmak için , video içindeki yüzü tespit ederiz
    #xml dosyası kullanılabilir hale getirildi
    face_detec = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    #frame i gri tona dönüştürdük
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces availale on cam, kaç tane yüz varsa videoda onların hepsini algılamayı sağlar
    # resimdeki yüzleri bulmak için detecMultiScale() kullaıldı
    num_faces = face_detec.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the cam and preprocess it
    # her bir yüze erişiyoruz 
    # her yüz için tanıma yapılır
    for (x, y, w, h) in num_faces:
    # x, y başlangıç boktası w genişlik h yükseklik
    # bu değerlere göre dikdmrtgen çızıyoryz her bir yüz için
        
        # yüzün çevresine dikdörtgen çizilir
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        
        # bulunan videodki yüz görüntülerini yüz üykseklik ve genişliğine göre kırpıyoruz ve roi ye depoluyoruz
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        
        #adlığımız kırptığımız her bir frame modelimiz için işlenmeli, modelimiz 48,48 boyutunda girdi alır
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions için kırpılmış işlenmiş gri kare üzernde yapılır

        emotion_prediction = emotion_model.predict(cropped_img)

        # modelimiz tahmin sonunda yüzde kaç kızgın, korkulu, mutlu yüzde değerleri alır bunun sonucunda biz max yüzde değerini alırız tahmin için
        maxIndex = int(np.argmax(emotion_prediction))

        cv2.putText(frame, emotion_classes[maxIndex], (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

