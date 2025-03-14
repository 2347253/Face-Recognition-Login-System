import cv2
import os
# import sqlite3
import numpy as np
from PIL import Image
# from FaceDetection.settings import BASE_DIR
import time

print('hi')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print (BASE_DIR)

detector = cv2.CascadeClassifier(BASE_DIR+'/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Create a connection witn databse
# conn = sqlite3.connect('db.sqlite3')
# if conn != 0:
#     print("Connection Successful")
# else:
#     print('Connection Failed')
#     exit()

# Creating table if it doesn't already exists
# conn.execute('''create table if not exists facedata ( id int primary key, name char(20) not null)''')

class FaceRecognition:    

    def faceDetect(self, Entry1,):
        face_id = Entry1
        cam = cv2.VideoCapture(0)
        
        count = 0
        while(True):

            ret, img = cam.read()
            # img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite(BASE_DIR+'/dataset/User.' + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('Register Face', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                break
    
        cam.release()   
        cv2.destroyAllWindows()

    
    def trainFace(self):
        # Path for face image database
        path = BASE_DIR+'/dataset'

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')

                face_id = int(os.path.split(imagePath)[-1].split(".")[1])
                print("face_id",face_id)
                faces = detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(face_id)

            return faceSamples,ids

        print ("\n Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.save(BASE_DIR+'/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))




    def recognizeFace(self):
        model_path = os.path.join(BASE_DIR, 'trainer/trainer.yml')
        cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

        # Check if the trained model exists
        if not os.path.exists(model_path):
            print("\n ❌ Error: Face recognition model not found! Train the model first.")
            return None

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)

        face_cascade = cv2.CascadeClassifier(cascade_path)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cam = cv2.VideoCapture(0)

        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        start_time = time.time()
        recognized_face_id = None  # Store recognized ID but don’t exit immediately

        while True:
            ret, img = cam.read()
            if not ret:
                print("\n ❌ Error: Failed to capture image from webcam.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            if len(faces) == 0:
                print("⚠️ No face detected. Please position your face properly.")

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                try:
                    face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                except Exception as e:
                    print(f"\n ❌ Error in prediction: {e}")
                    continue

                if confidence < 50:  # Face recognized
                    name = f"Detected (ID: {face_id})"
                    print(f"\n ✅ Face Recognized! ID: {face_id} | Confidence: {confidence:.2f}")
                    recognized_face_id = face_id  # Store the ID but don't exit loop

                else:  # Unknown face
                    name = "User Not Recognized"
                    print(f"\n ❌ Unknown User | Confidence: {confidence:.2f}")

                cv2.putText(img, name, (x + 5, y - 5), font, 1, (0, 255, 0) if confidence < 50 else (0, 0, 255), 2)
                cv2.putText(img, f"{confidence:.2f}", (x + 5, y + h - 5), font, 1, (0, 255, 0) if confidence < 50 else (0, 0, 255), 1)

            cv2.imshow('Detect Face', img)

            # Stop the loop after 15 seconds
            if time.time() - start_time > 15:
                print("\n ⏳ Time limit reached.")
                break

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' to exit manually
            if k == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Return the recognized ID if a face was detected
        return recognized_face_id if recognized_face_id is not None else None

