
import sys
import os
from detectors import FaceDetector
import otherop as oo
import cv2

def convert(people_folder):
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print("Have you added at least one person to the system?")
        sys.exit()
    print("This are the people in the Recognition System:")
    for person in people:
        print("-" + person)
    save_folder = "./gray"
    
    
    #video = VideoCamera()
    print(save_folder)
    detector = FaceDetector()
    counter = 1
    images = []
    labels = []
    
    for i, person in enumerate(people):
    	folder = save_folder +  '/' + person
    	os.mkdir(folder)
    	for image in os.listdir(people_folder + '/' + person):
        	frame  = cv2.imread(people_folder +'/'+ person + '/' + image, 0)
        	face_coord = detector.detect(frame)
        	if len(face_coord):
        		frame, face_img = oo.get_images(frame, face_coord)
        		cv2.imwrite(folder + '/' + str(counter) + '.jpg',face_img[0])
        		print('Images Saved:' + str(counter))
        		counter += 1
        		cv2.imshow('Saved Face', face_img[0])
        	cv2.waitKey(50)

if __name__ == '__main__':
	PEOPLE_FOLDER = "./people"
	convert(PEOPLE_FOLDER)