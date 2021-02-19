import cv2
import numpy as np


PATH_DIRECTORY = '/Users/sebastianrestrepo/Documents/AI/Camera_calculator/'

#video = cv2.VideoCapture(PATH_DIRECTORY + 'Test_videos/test1.avi')
video = cv2.VideoCapture(PATH_DIRECTORY + 'Test_videos/IMG_2450.mov')

#video = cv2.VideoCapture(0)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
image_center = (int(width / 2), int(height / 2))
rect_coord1 = (int(width/3), int(height / 3))
rect_coord2 = (int(width * 3/4.5), int(height * 2 / 3))
delta = 5



# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('/Users/sebastianrestrepo/Documents/AI/Camera_calculator/Test_videos/test1.avi',fourcc, 20.0, size)

if video.isOpened():
    while True:
        check, frame = video.read()
        if check:    
            # write the flipped frame
            #frame = cv2.flip(frame, 0)

            # Select roi from frame. Delta is to avoid selecting the displaying rectangle as a contour  
            image = frame[rect_coord1[1] + delta :rect_coord2[1]- delta, rect_coord1[0]+ delta:rect_coord2[0] - delta]
            

            cv2.rectangle(frame, rect_coord1 , rect_coord2, (80, 205, 255), 4) 


            # grayscale
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # Blur image then find edges using Canny 
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Binary otsu threshold https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
     


            #ret, binary = cv2.threshold(blurred ,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY, 11, 2)


            edged = cv2.Canny(binary, 2, 90)  # https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
            # canny is an edge detection algorithm based on the gradients of the image
            # Fint Contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) < 100 and len(contours) > 3 :
                cnts_rect = [cv2.boundingRect(c)  for c in contours]
                #rect_arr = np.asarray(cnts_rect)

                #varr = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] > 30]

                rectangles, weights = cv2.groupRectangles(cnts_rect + cnts_rect, groupThreshold=1, eps=0.2)




                for (x, y, w, h) in rectangles:
                    if w * h > 100 :
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2) 
                        cv2.imshow("Contours", binary)
                        cv2.imshow("Contours2", edged)




            

            #out.write(frame)

            cv2.imshow('Color Frame', frame)
            
            key = cv2.waitKey(50)
            if key == ord('q'):
                break

        else:
            print('Frame not available')
            print(video.isOpened())
            break




else:
    print("video not opened")

video.release()
#out.release()
cv2.destroyAllWindows()