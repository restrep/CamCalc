import cv2
import numpy as np
from functions import sort_contours_x_cord, relevant_rectangles



######## KNN block#############
# Let's take a look at our digits dataset
image = cv2.imread('digits.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)

# Split the image to 5000 cells, each 20x20 size
# This gives us a 4-dim array: 50 x 100 x 20 x 20
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Convert the List data type to Numpy Array of shape (50,100,20,20)
x = np.array(cells)
print ("The shape of our cells array: " + str(x.shape))

# Split the full data set into two segments
# One will be used fro Training the model, the other as a test data set
train = x[:,:70].reshape(-1,400).astype(np.float32) # Size = (3500,400)
test = x[:,70:100].reshape(-1,400).astype(np.float32) # Size = (1500,400)

# Create labels for train and test data
k = [0,1,2,3,4,5,6,7,8,9]
train_labels = np.repeat(k,350)[:,np.newaxis]
test_labels = np.repeat(k,150)[:,np.newaxis]

# Initiate kNN, train the data, then test it with test data for k=3
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE,train_labels)
ret, result, neighbors, distance = knn.findNearest(test, k=3)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * (100.0 / result.size)
print("Accuracy is = %.2f" % accuracy + "%")


##################################
PATH_DIRECTORY = '/Users/sebastianrestrepo/Documents/AI/Camera_calculator/'

#video = cv2.VideoCapture(0)
#video = cv2.VideoCapture(PATH_DIRECTORY + 'Test_videos/test1.avi')
video = cv2.VideoCapture(PATH_DIRECTORY + 'Test_videos/IMG_2450.mov')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
image_center = (int(width / 2), int(height / 2))
rect_coord1 = (int(width/3), int(height / 3))
rect_coord2 = (int(width * 3/4), int(height * 2 / 3))
delta = 6
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter(PATH_DIRECTORY + 'video_output.avi',fourcc, 10.0, size)

if video.isOpened():
    while True:
        check, frame = video.read()
        if check:    
           
            # Select roi from frame. Delta is to avoid selecting the displaying rectangle as a contour    
            image = frame[rect_coord1[1] + delta :rect_coord2[1]+ delta, rect_coord1[0]+ delta:rect_coord2[0]+ delta]
            

            cv2.rectangle(frame, rect_coord1, rect_coord2, (80, 205, 255), 4) 

            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # Blur image then find edges using Canny 
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY, 11, 2)
            edged = cv2.Canny(binary, 2, 90)  # https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
            # canny is an edge detection algorithm based on the gradients of the image

            # Find Contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) < 50 and len(contours) > 3 :
                # sort contours by position
                cnts_x_cord, cnts = sort_contours_x_cord(contours)

                rect_list = relevant_rectangles(cnts)
                #print(rect_list)
                numbers_list = []
                for (x, y, w, h) in rect_list:
                    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
                    if w >= 5 and h >= 5:
                        roi = blurred[y:y + h, x:x + w]
                        #cv2.imshow("blurred", blurred)
                        #cv2.imshow("roi", roi)
                        #print(np.mean(roi))
                        #final2 = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                        ret, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                        #cv2.imshow("roi2", roi)
                        final = cv2.resize(roi,(20,20), interpolation=cv2.INTER_AREA)
                        
                        #cv2.imshow("final", final)
                        #cv2.imshow("final2", final2)
                        #cv2.waitKey(0)
                        final_array = final.reshape((1,400))
                        final_array = final_array.astype(np.float32)
                        ret, result, neighbours, dist = knn.findNearest(final_array, k=3)
                    
                        
                        number = str(int(float(result[0])))
                        # draw a rectangle around the digit, the show what the
                        # digit was classified as
                        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.rectangle(frame, (x+rect_coord1[0], y+rect_coord1[1]), (x+rect_coord1[0]+w, y+rect_coord1[1]+h), (0, 255, 0), 2) 
                        cv2.putText(frame, number, (x + rect_coord1[0] , rect_coord1[1] + 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
                        #cv2.imshow("Color Frame", image)
                        numbers_list.append(float(number))

                suma = "=" + str(sum(numbers_list))
                cv2.putText(frame, suma, (rect_coord2[0] + 10 , rect_coord1[1] + 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
                #cv2.putText(frame, number, (x + rect_coord1[0] , rect_coord1[1] + 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)    
                    
                    
                    
                    #if w > 5 and h > 20:
                    #    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3) 
                    #    cv2.imshow("Color Frame", image)



    
            cv2.imshow('Color Frame', frame)
            cv2.imshow('Color Frame 2', edged)
            key = cv2.waitKey(50)

             # write the frame
            #out.write(frame)

            if key == ord('q'):
                break
        else:
            print('Frame not available')
            print(video.isOpened())
else:
    print("video not opened")

video.release()
#out.release()
cv2.destroyAllWindows()