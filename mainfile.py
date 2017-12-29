"""

main

"""
import sys
import os
import numpy as np
from videocamera import VideoCamera
from detectors import FaceDetector
import otherop as oo

import cv2
import time


def add_person(people_folder):
    person_name = input('What is the name of the new person: ').lower()
    folder = people_folder +  '/' + person_name
    if not os.path.exists(folder):
        input("I will now take 20 pictures. Press ENTER when ready.")
        os.mkdir(folder)
        video = VideoCamera()
        detector = FaceDetector()
        counter = 1
        timer = 0
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
        while counter < 21:
            frame = video.get_frame()
            face_coord = detector.detect(frame)
            if len(face_coord):
                frame, face_img = oo.get_images(frame, face_coord)
                
                if timer % 100 == 5:
                    cv2.imwrite(folder + '/' + str(counter) + '.jpg',
                                face_img[0])
                    print('Images Saved:' + str(counter))
                    counter += 1
                    cv2.imshow('Saved Face', face_img[0])

            cv2.imshow('Video Feed', frame)
            cv2.waitKey(50)
            timer += 5
    else:
        print("This name already exists.")
        sys.exit()

def recognize_people(people_folder):
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print("Have you added at least one person to the system?")
        sys.exit()
    print("This are the people in the Recognition System:")
    for person in people:
        print("-" + person)

    start = time.time()
    #choice = check_choice()
    recognizer = None
    detector = FaceDetector()
    
    recognizer = cv2.face.createLBPHFaceRecognizer()
    threshold = 91 #93
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + '/' + person):
            images.append(cv2.imread(people_folder +'/'+ person + '/' + image, 0))
            labels.append(i)
    try:
        recognizer.train(images, np.array(labels))
        print("train model")
    except:
        print("\nOpenCV Error: Do you have at least two people in the database?\n")
        sys.exit()
    end = time.time()
    print(end - start)
    video = VideoCamera()
    while True:
        frame = video.get_frame()
        faces_coord = detector.detect(frame)
        if len(faces_coord):
            frame, faces_img = oo.get_images(frame, faces_coord)
            for i, face_img in enumerate(faces_img):
                if __version__ == "3.1.0":
                    collector = cv2.face.MinDistancePredictCollector()
                    recognizer.predict(face_img, collector)
                    conf = collector.getDist()
                    pred = collector.getLabel()
                else:
                    pred, conf = recognizer.predict(face_img)
                print("Prediction: " + str(pred))
                print('Confidence: ' + str(round(conf)))
                print('Threshold: ' + str(threshold))
                if conf < threshold:
                    cv2.putText(frame, labels_people[pred].capitalize(),
                                (faces_coord[i][0], faces_coord[i][1] - 2),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                                cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Unknown",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (206, 0, 209), 2,
                                cv2.LINE_AA)

        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (206, 0, 209), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == 27:
            sys.exit()



def check_choice():
    
    is_valid = 0
    while not is_valid:
        try:
            choice = int(input('Enter your choice [1-3] : '))
            if choice in [1, 2, 3]:
                is_valid = 1
            else:
                print("'%d' is not an option.\n" % choice)
        except error:
            print("%s is not an option.\n" % str(error).split(": ")[1])
    return choice

if __name__ == '__main__':
    print(30 * '=-')
    print("   POSSIBLE ACTIONS")
    print(40 * '*')
    print("1. Add person to the recognizer system")
    print("2. Start recognizer")
    print("3. Exit")
    print(40 * '*')

    CHOICE = check_choice()

    PEOPLE_FOLDER = "/home/parth/Desktop/internal/people"
    #SHAPE = "ellipse"

    if CHOICE == 1:
        if not os.path.exists(PEOPLE_FOLDER):
            os.makedirs(PEOPLE_FOLDER)
        add_person(PEOPLE_FOLDER)
    elif CHOICE == 2:
        recognize_people(PEOPLE_FOLDER)
    elif CHOICE == 3:
        sys.exit()

    
