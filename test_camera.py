import cv2


video = cv2.VideoCapture(0)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
image_center = (int(width / 2), int(height / 2))
rect_coord1 = (int(width/4), int(height / 3))
rect_coord2 = (int(width * 3/4), int(height * 2 / 3))
delta = 6



# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('/Users/sebastianrestrepo/Documents/AI/Camera_calculator/Test_videos/test1.avi',fourcc, 20.0, size)

if video.isOpened():
    while True:
        check, frame = video.read()
        if check:    
            # write the flipped frame
            #frame = cv2.flip(frame, 0)

            # Select roi from frame. Delta is to avoid selecting the displaying rectangle as a contour  
            image = frame[rect_coord1[1] + delta :rect_coord2[1]+ delta, rect_coord1[0]+ delta:rect_coord2[0]+ delta]
            

            cv2.rectangle(frame, rect_coord1, rect_coord2, (80, 205, 255), 4) 

            out.write(frame)

            cv2.imshow('Color Frame', frame)
            key = cv2.waitKey(50)
            if key == ord('q'):
                break
        else:
            print('Frame not available')
            print(video.isOpened())
else:
    print("video not opened")

video.release()
out.release()
cv2.destroyAllWindows()