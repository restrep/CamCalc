#!/usr/bin/env python
# coding: utf-8

# ## Mini Project # 9 - Handwritten Digit Recognition

# ### Data Prep, Training and Evaluation

# ### Defining some functions we will use to prepare an input image

# In[1]:


import numpy as np
import cv2

# Define our functions

def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    
    #if cv2.contourArea(contour) > 10: # units are in pixels
    M = cv2.moments(contour)    # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
    #print(M)
    if M['m00'] == 0:
        return 0
    else:
        cx = int(M['m10']/M['m00']) # x centroid.  
        return (cx)

def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed
    
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        #print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = int((width - height)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,                                                   pad,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad = int((width - height)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,                                                   cv2.BORDER_CONSTANT,value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square


def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions
    # The desired dimensions depend on the model input
    
    buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg


# In[1]:


import numpy as np
import cv2


PATH_DIRECTORY = '/Users/sebastianrestrepo/Documents/AI/Camera_calculator/'

WindowName="image"
view_window = cv2.namedWindow(WindowName,cv2.WINDOW_NORMAL)

# These two lines will force the window to be on top with focus.
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)


image = cv2.imread(PATH_DIRECTORY + 'Images/3.png')
#image = cv2.imread(PATH_DIRECTORY + 'Images/numbers1.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.imshow("gray", gray)
cv2.waitKey(0)

# Blur image then find edges using Canny 
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)


threshold = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)

threshold2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY, 11, 2)

cv2.imshow("adaptive", threshold)
cv2.imshow("adaptive2", threshold2)
cv2.waitKey(0)

# Binary otsu threshold https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
ret, binary = cv2.threshold(blurred ,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("binary", binary)
cv2.waitKey(0)
print(ret)

edged = cv2.Canny(threshold, 2, 90)  # https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
# canny is an edge detection algorithm based on the gradients of the image
cv2.imshow("edged", edged)
cv2.waitKey(0)



# Fint Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours,-1,(0,255,0),3)
cv2.imshow('contours',image)


cnts_rect = [cv2.boundingRect(c) for c in contours]

rect_arr = np.asarray(cnts_rect)



print(cnts_rect)
print('------')


rectangles, weights = cv2.groupRectangles(cnts_rect + cnts_rect, groupThreshold=1, eps=0.2)
print(rectangles)

for (x, y, w, h) in rectangles:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2) 
    cv2.imshow("Contours", image)
    
    
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)


# ## Loading a new image, preprocessing it and classifying the digits

# In[2]:


display(cnts_rect)
print(rectangles)
print('--------')
#rectan, weight = cv2.groupRectangles(rectangles + rectangles, groupThreshold=1, eps=0.2)
#print(rectan)


# In[12]:


rectangles2 = [(x, y, w, h) for (x, y, w, h) in rectangles]
display(rectangles2 + rectangles2)
rectan, weight = cv2.groupRectangles(rectangles2 + rectangles2, groupThreshold=1, eps=0.2)
print(rectan)


# In[2]:


import numpy as np
import cv2



WindowName="image"
view_window = cv2.namedWindow(WindowName,cv2.WINDOW_NORMAL)

# These two lines will force the window to be on top with focus.
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)



image = cv2.imread('Images/1.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image", image)
cv2.imshow("gray", gray)
#cv2.waitKey(0)

# Blur image then find edges using Canny 
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imshow("blurred", blurred)
#cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)  # https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
# canny is an edge detection algorithm based on the gradients of the image
cv2.imshow("edged", edged)
#cv2.waitKey(0)


# Fint Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# In OpenCV, finding contours is like finding white object from black background.
# So remember, object to be found should be white and background should be black.

# cv2.CHAIN_APPROX_SIMPLE is the contourapprox method: how many points of the contour are stored
# cv2.RETR_EXTERNAL: contour retrieval mode. Depending on the image contours can form a hierarchy, decides how to extract that hierarchy. 
# External means just the first hierarchy


#print(hierarchy)
#Sort out contours left to right by using their x cordinates
#contours = sorted(contours, key = x_cord_contour, reverse = False)

print(len(contours))

def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    M = cv2.moments(contour)    # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
    #print(M)
 #   if M['m00'] == 0:
 #       return 0
 #   else:
 #       cx = int(M['m10']/M['m00']) # x centroid.  
 #       return (cx)
    (x, y, w, h) = cv2.boundingRect(contour)  # x,y refers to top left corner
    return x
    
    
def sort_contours_size(cnts):
    """ Sort contours based on the size"""
    cnts_sizes = [cv2.contourArea(contour) for contour in cnts if cv2.contourArea(contour) > 0]
    (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, cnts), key=lambda tup: tup[0], reverse=True))
    return cnts_sizes, cnts

def sort_contours_x_cord(cnts):
    """ Sort contours based on the size"""
    cnts_x_cord = [x_cord_contour(contour) for contour in cnts if x_cord_contour(contour) > 0]
    (cnts_x_cord, cnts) = zip(*sorted(zip(cnts_x_cord, cnts), key=lambda tup: tup[0], reverse=False))
    return cnts_x_cord, cnts



cnts_x_cord, cnts = sort_contours_x_cord(contours)

cnts_sizes, cnts2 = sort_contours_size(contours)



cnts_rect = [cv2.boundingRect(contour) for contour in cnts]
rect_arr = np.asarray(cnts_rect)


print(cnts_x_cord)
print(cnts_sizes)

def doOverlap(xl1,yl1, xr1,yr1, xl2,yl2, xr2,yr2): 
    """ l1: Top Left coordinate of first rectangle.
r1: Bottom Right coordinate of first rectangle.
l2: Top Left coordinate of second rectangle.
r2: Bottom Right coordinate of second rectangle."""
      
    # If one rectangle is on left side of other 
    if(xl1 >= xr2 or xl2 >= xr1): 
        return False
  
    # If one rectangle is above other 
    if(yl1 <= yr2 or yl2 <= yr1): 
        return False
  
    return True

# Create empty array to store entire number
final_cnts = []
(x1, y1, w1, h1) = cv2.boundingRect(cnts[0]) 
vertical_pad = int(max(rect_arr[:, 3]) / 2)


for index, c in enumerate(cnts[:-1]):
#    print(index)
#    print(cv2.contourArea(c))
    
    #(x1, y1, w1, h1) = cv2.boundingRect(c) 
    xl1,yl1, xr1, yr1 = x1, y1+h1, x1+w1, y1
#    cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1 + vertical_pad), (0, 255, 0), 2)
#    print('green')
#    print(x1, y1, w1, h1)
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    (x2, y2, w2, h2) = cv2.boundingRect(cnts[index+1])
    xl2, yl2, xr2, yr2 = x2, y2+h2, x2+w2, y2
#    cv2.rectangle(image, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
#    print(x2, y2, w2, h2)
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    
    inter = doOverlap(xl1, yl1 + vertical_pad, xr1,yr1, xl2,yl2, xr2,yr2)
#    print(index)
#    print(final_cnts)
    if index == 0:
        final_cnts.append((x1, y1, w1, h1))
#        print('continue')
        (x1, y1, w1, h1) = cv2.boundingRect(cnts[1]) 
        continue
        
    if inter==True and index > 0:
        x1 = min(xl1, xl2)
        y1 = min(yr1, yr2)
        w1 = max(xr1, xr2) - x1
        h1 = max(yl1, yl2) - y1
        
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
#        print(x1, y1, w1, h1)
#        print('red')
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        
        # check with previous on the list
        xprev, yprev, wprev, hprev = final_cnts[len(final_cnts) - 1]
        xlprev,ylprev, xrprev, yrprev = xprev, yprev+hprev, xprev+wprev, yprev
        
        xlcurrent,ylcurrent, xrcurrent, yrcurrent = x1, y1+h1, x1+w1, y1
        
        inter_previous = doOverlap(xlcurrent, ylcurrent + vertical_pad, xrcurrent, yrcurrent, xlprev,ylprev, xrprev, yrprev)
        
        if inter_previous == True:
            x1 = min(xlprev, xlcurrent)
            y1 = min(yrprev, yrcurrent)
            w1 = max(xrprev, xrcurrent) - x1
            h1 = max(ylprev, ylcurrent) - y1
            
            final_cnts = final_cnts[:-1]    
        final_cnts.append((x1, y1, w1, h1))
        #final_cnts.append((x1, y1, w1, h1))
        
    else:
        (x1, y1, w1, h1) = cv2.boundingRect(cnts[index + 1]) 
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
#        print('blue')
#        print(x1, y1, w1, h1)
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        
        
        # check with previous on the list
        xprev, yprev, wprev, hprev = final_cnts[len(final_cnts) - 1]
        xlprev,ylprev, xrprev, yrprev = xprev, yprev+hprev, xprev+wprev, yprev
        
        xlcurrent,ylcurrent, xrcurrent, yrcurrent = x1, y1+h1, x1+w1, y1
        
        inter_previous = doOverlap(xlcurrent, ylcurrent + vertical_pad, xrcurrent, yrcurrent, xlprev,ylprev, xrprev, yrprev)
        
        if inter_previous == True:
            x1 = min(xlprev, xlcurrent)
            y1 = min(yrprev, yrcurrent)
            w1 = max(xrprev, xrcurrent) - x1
            h1 = max(ylprev, ylcurrent) - y1
            final_cnts = final_cnts[:-1]  
            
        final_cnts.append((x1, y1, w1, h1))
    
    
   
    
#print(final_cnts)

rectangles, weights = cv2.groupRectangles(final_cnts + final_cnts, groupThreshold=1, eps=0.2)
rectangles2, weights = cv2.groupRectangles(cnts_rect + cnts_rect, groupThreshold=1, eps=0.2)

#print(rectangles)
#print(rectangles2)

# loop over the contours
#for (x, y, w, h) in final_cnts:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 166), 2) 
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    
#cv2.waitKey(0)    
  
#for c in cnts:
#    # compute the bounding box for the rectangle
#    (x, y, w, h) = cv2.boundingRect(c)    
#        
#    
#    cv2.drawContours(image, contours, -1, (0,255,0), 2)
#    
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#    
#    print(x,y,w,h)
#    
#    cv2.imshow("Contours", image)
#    #cv2.waitKey(0)

for (x, y, w, h) in final_cnts:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2) 
    cv2.imshow("Contours", image)

    
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)
#print ("The number is: " + ''.join(full_number))
print(len(final_cnts))
print(len(rectangles))
print(len(rectangles2))
print(len(cnts))


# In[7]:


print(final_cnts)
print(rectangles)
print(cnts_sizes)
np.mean(cnts_sizes)
np.sqrt(np.mean(cnts_sizes))


# In[5]:


cnts_height = [cv2.boundingRect(contour) for contour in cnts]
print(cnts_height)
arr = np.asarray(cnts_height)
print(arr)
0.5 * max(arr[:, 3])


# In[ ]:


if w >= 5 and h >= 25:
       roi = blurred[y:y + h, x:x + w]
       ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
       squared = makeSquare(roi)
       final = resize_to_pixel(20, squared)
       cv2.imshow("final", final)
       final_array = final.reshape((1,400))
       final_array = final_array.astype(np.float32)
       ret, result, neighbours, dist = knn.find_nearest(final_array, k=1)
       number = str(int(float(result[0])))
       full_number.append(number)
       # draw a rectangle around the digit, the show what the
       # digit was classified as
       cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
       cv2.putText(image, number, (x , y + 155),
           cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
       cv2.imshow("image", image)
       cv2.waitKey(0) 
       


# In[3]:


print(image.shape)


# In[1]:


import numpy as np
import cv2



WindowName="image"
view_window = cv2.namedWindow(WindowName,cv2.WINDOW_NORMAL)

# These two lines will force the window to be on top with focus.
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)



image = cv2.imread('images/numbers8.jpg')



grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayimage,(5,5),0)
cv2.imshow("Image", image)
ret, im_th = cv2.threshold(grayimage, 90, 255, cv2.THRESH_BINARY_INV)
#edged = cv2.Canny(blurred, 30, 150)
#cv2.imshow("Edges", edged)
(cnts, _) = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


print(len(cnts))

cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key =lambda x: x[1])
for (c, _) in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	if w >= 6 and h >= 10:
		cv2.rectangle(image, (x-6, y-6), (x + w+6, y + h+6),(0, 255, 0), 1)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


import numpy as np
import cv2



WindowName="image"
view_window = cv2.namedWindow(WindowName,cv2.WINDOW_NORMAL)

# These two lines will force the window to be on top with focus.
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)



image = cv2.imread('images/numbers5.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image", image)
cv2.imshow("gray", gray)
#cv2.waitKey(0)

# Blur image then find edges using Canny 
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imshow("blurred", blurred)
#cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)  # https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
# canny is an edge detection algorithm based on the gradients of the image
cv2.imshow("edged", edged)
#cv2.waitKey(0)


# Fint Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# In OpenCV, finding contours is like finding white object from black background.
# So remember, object to be found should be white and background should be black.

# cv2.CHAIN_APPROX_SIMPLE is the contourapprox method: how many points of the contour are stored
# cv2.RETR_EXTERNAL: contour retrieval mode. Depending on the image contours can form a hierarchy, decides how to extract that hierarchy. 
# External means just the first hierarchy


#print(hierarchy)
#Sort out contours left to right by using their x cordinates
#contours = sorted(contours, key = x_cord_contour, reverse = False)

print(len(contours))

def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    M = cv2.moments(contour)    # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
    #print(M)
 #   if M['m00'] == 0:
 #       return 0
 #   else:
 #       cx = int(M['m10']/M['m00']) # x centroid.  
 #       return (cx)
    (x, y, w, h) = cv2.boundingRect(contour)  # x,y refers to top left corner
    return x
    
    
def sort_contours_size(cnts):
    """ Sort contours based on the size"""
    cnts_sizes = [cv2.contourArea(contour) for contour in cnts if cv2.contourArea(contour) > 0]
    (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, cnts), key=lambda tup: tup[0], reverse=True))
    return cnts_sizes, cnts

def sort_contours_x_cord(cnts):
    """ Sort contours based on the size"""
    cnts_x_cord = [x_cord_contour(contour) for contour in cnts if x_cord_contour(contour) > 0]
    (cnts_x_cord, cnts) = zip(*sorted(zip(cnts_x_cord, cnts), key=lambda tup: tup[0], reverse=False))
    return cnts_x_cord, cnts



cnts_x_cord, cnts = sort_contours_x_cord(contours)

cnts_sizes, cnts2 = sort_contours_size(contours)



cnts_rect = [cv2.boundingRect(contour) for contour in cnts]
rect_arr = np.asarray(cnts_rect)


print(cnts_x_cord)
print(cnts_sizes)

def doOverlap(xl1,yl1, xr1,yr1, xl2,yl2, xr2,yr2): 
    """ l1: Top Left coordinate of first rectangle.
r1: Bottom Right coordinate of first rectangle.
l2: Top Left coordinate of second rectangle.
r2: Bottom Right coordinate of second rectangle."""
      
    # If one rectangle is on left side of other 
    if(xl1 >= xr2 or xl2 >= xr1): 
        return False
  
    # If one rectangle is above other 
    if(yl1 <= yr2 or yl2 <= yr1): 
        return False
  
    return True

# Create empty array to store entire number
final_cnts = []
(x1, y1, w1, h1) = cv2.boundingRect(cnts[0]) 
vertical_pad = int(max(rect_arr[:, 3]) / 2)


for index, c in enumerate(cnts[:-1]):
#    print(index)
#    print(cv2.contourArea(c))
    
    (x1, y1, w1, h1) = cv2.boundingRect(cnts[index]) 
    xl1,yl1, xr1, yr1 = x1, y1+h1, x1+w1, y1
#    cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1 ), (0, 255, 0), 2)
#    print('green')
#    print(x1, y1, w1, h1)
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    (x2, y2, w2, h2) = cv2.boundingRect(cnts[index+1])
    xl2, yl2, xr2, yr2 = x2, y2+h2, x2+w2, y2
#    cv2.rectangle(image, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
#    print(x2, y2, w2, h2)
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    
    inter = doOverlap(xl1, yl1 + vertical_pad, xr1, yr1, xl2, yl2 + vertical_pad, xr2,yr2)
 #   print(index)
 #   print(final_cnts)
#    if index == 0:
#        final_cnts.append((x1, y1, w1, h1))
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 200, 200), 2)
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
#       print('continue')
#       (x1, y1, w1, h1) = cv2.boundingRect(cnts[1]) 
#       continue
        
    if inter==True:
        x1 = min(xl1, xl2)
        y1 = min(yr1, yr2)
        w1 = max(xr1, xr2) - x1
        h1 = max(yl1, yl2) - y1
        
 #       cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
 #       print(x1, y1, w1, h1)
 #       print('red')
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        
        if index > 0:
            # check with previous on the list
            xprev, yprev, wprev, hprev = final_cnts[len(final_cnts) - 1]
            xlprev,ylprev, xrprev, yrprev = xprev, yprev+hprev, xprev+wprev, yprev

            xlcurrent,ylcurrent, xrcurrent, yrcurrent = x1, y1+h1, x1+w1, y1

            inter_previous = doOverlap(xlcurrent, ylcurrent + vertical_pad, xrcurrent, yrcurrent, xlprev,ylprev, xrprev, yrprev)

            if inter_previous == True:
                x1 = min(xlprev, xlcurrent)
                y1 = min(yrprev, yrcurrent)
                w1 = max(xrprev, xrcurrent) - x1
                h1 = max(ylprev, ylcurrent) - y1

                final_cnts = final_cnts[:-1]    
        final_cnts.append((x1, y1, w1, h1))
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 200, 200), 2)
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        #final_cnts.append((x1, y1, w1, h1))
        
    else:
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 200, 0), 2)
        (x1, y1, w1, h1) = cv2.boundingRect(cnts[index + 0]) 
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
#        print('blue')
#        print(x1, y1, w1, h1)
#       cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        
        
        # check with previous on the list
        xprev, yprev, wprev, hprev = final_cnts[len(final_cnts) - 1]
        xlprev,ylprev, xrprev, yrprev = xprev, yprev+hprev, xprev+wprev, yprev
        
        xlcurrent,ylcurrent, xrcurrent, yrcurrent = x1, y1+h1, x1+w1, y1
        
        inter_previous = doOverlap(xlcurrent, ylcurrent + vertical_pad, xrcurrent, yrcurrent, xlprev,ylprev, xrprev, yrprev)
        
        if inter_previous == True:
            x1 = min(xlprev, xlcurrent)
            y1 = min(yrprev, yrcurrent)
            w1 = max(xrprev, xrcurrent) - x1
            h1 = max(ylprev, ylcurrent) - y1
            final_cnts = final_cnts[:-1]  
            
        final_cnts.append((x1, y1, w1, h1))
        
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 200, 200), 2)
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        
        if index == len(cnts) - 2:
            (x1, y1, w1, h1) = cv2.boundingRect(cnts[index + 1])
            final_cnts.append((x1, y1, w1, h1))
            
    
        
        
    
    
   
    
#print(final_cnts)

rectangles, weights = cv2.groupRectangles(final_cnts + final_cnts, groupThreshold=1, eps=0.2)
rectangles2, weights = cv2.groupRectangles(cnts_rect + cnts_rect, groupThreshold=1, eps=0.2)

#print(rectangles)
#print(rectangles2)

# loop over the contours
#for (x, y, w, h) in final_cnts:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 166), 2) 
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    
#cv2.waitKey(0)    
  
#for c in cnts:
#    # compute the bounding box for the rectangle
#    (x, y, w, h) = cv2.boundingRect(c)    
#        
#    
#    cv2.drawContours(image, contours, -1, (0,255,0), 2)
#    
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#    
#    print(x,y,w,h)
#    
#    cv2.imshow("Contours", image)
#    #cv2.waitKey(0)
full_number = []
for (x, y, w, h) in final_cnts:
    
    if w >= 5 and h >= 20:
        roi = blurred[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
        final = cv2.resize(roi,(20,20))
        cv2.imshow("final", final)
        cv2.waitKey(0)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
    
        
        number = str(int(float(result[0])))
        full_number.append(number)
        # draw a rectangle around the digit, the show what the
        # digit was classified as
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, number, (x , y + 2 * vertical_pad),
            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)
    
    
    
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2) 
    cv2.imshow("Contours", image)

    
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)
#print ("The number is: " + ''.join(full_number))
print(len(final_cnts))
print(len(rectangles))
print(len(rectangles2))
print(len(cnts))


# In[ ]:


import numpy as np
import cv2



WindowName="image"
view_window = cv2.namedWindow(WindowName,cv2.WINDOW_NORMAL)

# These two lines will force the window to be on top with focus.
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)



#image = cv2.imread('images/numbers.jpg')
image = cv2.imread('images/numbers_1.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image", image)
cv2.imshow("gray", gray)
#cv2.waitKey(0)

# Blur image then find edges using Canny 
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow("blurred", blurred)
#cv2.waitKey(0)

#edged = cv2.Canny(blurred, 30, 150)  # https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
edged = cv2.Canny(blurred, 1, 20)
# canny is an edge detection algorithm based on the gradients of the image
cv2.imshow("edged", edged)
cv2.waitKey(0)


# Fint Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# In OpenCV, finding contours is like finding white object from black background.
# So remember, object to be found should be white and background should be black.

# cv2.CHAIN_APPROX_SIMPLE is the contourapprox method: how many points of the contour are stored
# cv2.RETR_EXTERNAL: contour retrieval mode. Depending on the image contours can form a hierarchy, decides how to extract that hierarchy. 
# External means just the first hierarchy


#print(hierarchy)
#Sort out contours left to right by using their x cordinates
#contours = sorted(contours, key = x_cord_contour, reverse = False)

print(len(contours))

def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    M = cv2.moments(contour)    # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
    #print(M)
 #   if M['m00'] == 0:
 #       return 0
 #   else:
 #       cx = int(M['m10']/M['m00']) # x centroid.  
 #       return (cx)
    (x, y, w, h) = cv2.boundingRect(contour)  # x,y refers to top left corner
    return x
    
    
def sort_contours_size(cnts):
    """ Sort contours based on the size"""
    cnts_sizes = [cv2.contourArea(contour) for contour in cnts if cv2.contourArea(contour) > 0]
    (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, cnts), key=lambda tup: tup[0], reverse=True))
    return cnts_sizes, cnts

def sort_contours_x_cord(cnts):
    """ Sort contours based on the size"""
    cnts_x_cord = [x_cord_contour(contour) for contour in cnts if x_cord_contour(contour) > 0]
    (cnts_x_cord, cnts) = zip(*sorted(zip(cnts_x_cord, cnts), key=lambda tup: tup[0], reverse=False))
    return cnts_x_cord, cnts



cnts_x_cord, cnts = sort_contours_x_cord(contours)

cnts_sizes, cnts2 = sort_contours_size(contours)



cnts_rect = [cv2.boundingRect(contour) for contour in cnts]
rect_arr = np.asarray(cnts_rect)


print(cnts_x_cord)
print(cnts_sizes)

def doOverlap(xl1,yl1, xr1,yr1, xl2,yl2, xr2,yr2): 
    """ l1: Top Left coordinate of first rectangle.
r1: Bottom Right coordinate of first rectangle.
l2: Top Left coordinate of second rectangle.
r2: Bottom Right coordinate of second rectangle."""
      
    # If one rectangle is on left side of other 
    if(xl1 >= xr2 or xl2 >= xr1): 
        return False
  
    # If one rectangle is above other 
    if(yl1 <= yr2 or yl2 <= yr1): 
        return False
  
    return True

# Create empty array to store entire number
final_cnts = []
(x1, y1, w1, h1) = cv2.boundingRect(cnts[0]) 
vertical_pad = 3 * int(max(rect_arr[:, 3]) / 2)


for index, c in enumerate(cnts[:-1]):
#    print(index)
#    print(cv2.contourArea(c))
    
    (x1, y1, w1, h1) = cv2.boundingRect(cnts[index]) 
    xl1,yl1, xr1, yr1 = x1, y1+h1, x1+w1, y1
#    cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1 ), (0, 255, 0), 2)
#    print('green')
#    print(x1, y1, w1, h1)
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    (x2, y2, w2, h2) = cv2.boundingRect(cnts[index+1])
    xl2, yl2, xr2, yr2 = x2, y2+h2, x2+w2, y2
#    cv2.rectangle(image, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
#    print(x2, y2, w2, h2)
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    
    inter = doOverlap(xl1, yl1 + vertical_pad, xr1, yr1, xl2, yl2 + vertical_pad, xr2,yr2)
 #   print(index)
 #   print(final_cnts)
#    if index == 0:
#        final_cnts.append((x1, y1, w1, h1))
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 200, 200), 2)
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
#       print('continue')
#       (x1, y1, w1, h1) = cv2.boundingRect(cnts[1]) 
#       continue
        
    if inter==True:
        x1 = min(xl1, xl2)
        y1 = min(yr1, yr2)
        w1 = max(xr1, xr2) - x1
        h1 = max(yl1, yl2) - y1
        
 #       cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
 #       print(x1, y1, w1, h1)
 #       print('red')
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        
        if index > 0:
            # check with previous on the list
            xprev, yprev, wprev, hprev = final_cnts[len(final_cnts) - 1]
            xlprev,ylprev, xrprev, yrprev = xprev, yprev+hprev, xprev+wprev, yprev

            xlcurrent,ylcurrent, xrcurrent, yrcurrent = x1, y1+h1, x1+w1, y1

            inter_previous = doOverlap(xlcurrent, ylcurrent + vertical_pad, xrcurrent, yrcurrent, xlprev,ylprev + vertical_pad, xrprev, yrprev)

            if inter_previous == True:
                x1 = min(xlprev, xlcurrent)
                y1 = min(yrprev, yrcurrent)
                w1 = max(xrprev, xrcurrent) - x1
                h1 = max(ylprev, ylcurrent) - y1

                final_cnts = final_cnts[:-1]    
        final_cnts.append((x1, y1, w1, h1))
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 200, 200), 2)
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        #final_cnts.append((x1, y1, w1, h1))
        
    else:
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 200, 0), 2)
        (x1, y1, w1, h1) = cv2.boundingRect(cnts[index + 0]) 
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
#        print('blue')
#        print(x1, y1, w1, h1)
#       cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        
        
        # check with previous on the list
        xprev, yprev, wprev, hprev = final_cnts[len(final_cnts) - 1]
        xlprev,ylprev, xrprev, yrprev = xprev, yprev+hprev, xprev+wprev, yprev
        
        xlcurrent,ylcurrent, xrcurrent, yrcurrent = x1, y1+h1, x1+w1, y1
        
        inter_previous = doOverlap(xlcurrent, ylcurrent + vertical_pad, xrcurrent, yrcurrent, xlprev,ylprev+ vertical_pad, xrprev, yrprev)
        
        if inter_previous == True:
            x1 = min(xlprev, xlcurrent)
            y1 = min(yrprev, yrcurrent)
            w1 = max(xrprev, xrcurrent) - x1
            h1 = max(ylprev, ylcurrent) - y1
            final_cnts = final_cnts[:-1]  
            
        final_cnts.append((x1, y1, w1, h1))
        
#        cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (255, 200, 200), 2)
#        cv2.imshow("Contours", image)
#        cv2.waitKey(0)
        
        if index == len(cnts) - 2:
            (x1, y1, w1, h1) = cv2.boundingRect(cnts[index + 1])
            final_cnts.append((x1, y1, w1, h1))
            
    
        
        
    
    
   
    
#print(final_cnts)

rectangles, weights = cv2.groupRectangles(final_cnts + final_cnts, groupThreshold=1, eps=0.2)
rectangles2, weights = cv2.groupRectangles(cnts_rect + cnts_rect, groupThreshold=1, eps=0.2)

#print(rectangles)
#print(rectangles2)

# loop over the contours
#for (x, y, w, h) in final_cnts:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 166), 2) 
#    cv2.imshow("Contours", image)
#    cv2.waitKey(0)
    
#cv2.waitKey(0)    
  
#for c in cnts:
#    # compute the bounding box for the rectangle
#    (x, y, w, h) = cv2.boundingRect(c)    
#        
#    
#    cv2.drawContours(image, contours, -1, (0,255,0), 2)
#    
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#    
#    print(x,y,w,h)
#    
#    cv2.imshow("Contours", image)
#    #cv2.waitKey(0)
full_number = []
for (x, y, w, h) in final_cnts:
    
    if w >= 5 and h >= 20:
        roi = blurred[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
        final = cv2.resize(roi,(20,20),interpolation=cv2.INTER_AREA)
        #final2 = cv2.resize(cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2), (20,20))
        cv2.imshow("final", final)
        #cv2.imshow("final2", final2)
        cv2.waitKey(0)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
    
        
        number = str(int(float(result[0])))
        full_number.append(number)
        # draw a rectangle around the digit, the show what the
        # digit was classified as
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2) 
        cv2.putText(image, number, (x , y + int(vertical_pad / 2)),
            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("Contours", image)
        cv2.waitKey(0)
    
    
    
    


    
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)
#print ("The number is: " + ''.join(full_number))
print(len(final_cnts))
print(len(rectangles))
print(len(rectangles2))
print(len(cnts))


# In[ ]:




