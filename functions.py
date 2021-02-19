import cv2
import numpy as np

def x_cord_contour(contour):
    """ X coordinate of bounding box of contour"""
     # x,y refers to top left corner
    (x, y, w, h) = cv2.boundingRect(contour)
    return x  
    
def sort_contours_size(cnts):
    """ Sort contours based on the size"""
    cnts_sizes = [cv2.contourArea(contour) for contour in cnts if cv2.contourArea(contour) > 0]
    (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, cnts), key=lambda tup: tup[0], reverse=True))
    return cnts_sizes, cnts

def sort_contours_x_cord(cnts):
    """ Sort contours based on their position"""
    cnts_x_cord = [x_cord_contour(contour) for contour in cnts if x_cord_contour(contour) > 0]
    (cnts_x_cord, cnts) = zip(*sorted(zip(cnts_x_cord, cnts), key=lambda tup: tup[0], reverse=False))
    return cnts_x_cord, cnts

def doOverlap(xl1, yl1, xr1, yr1, xl2, yl2, xr2, yr2): 
    """ Boolean for intersecting rectangles
    l1: Top Left coordinate of first rectangle.
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


def merge_overlapping_rectangles(xl1, yl1, xr1, yr1, xl2, yl2, xr2, yr2):
    """Returns the coordinates (botton left x,y)  and w, h of merged rectangle """
    inter = doOverlap(xl1, yl1, xr1, yr1, xl2, yl2, xr2, yr2)
    x = min(xl1, xl2)
    y = min(yr1, yr2)
    w = max(xr1, xr2) - x
    h = max(yl1, yl2) - y
    return x, y, w, h


def relevant_rectangles(contours):

    # Create empty array to store final rectangles
    rectangles = []
    # Sort by position
    cnts_x_cord, cnts = sort_contours_x_cord(contours)
    # Add a vertical pad to secure overlap of some rectangles with similar x coord.
    # The padding is chosen as half the maximum rectangle found
    rect_array = np.asarray([cv2.boundingRect(contour) for contour in cnts])
    vertical_pad = int(max(rect_array[:, 3]) / 2)

    for index, c in enumerate(cnts[:-1]):

        # Coordinates of current rectangle. 
        (x1, y1, w1, h1) = cv2.boundingRect(cnts[index]) 
        xl1, yl1, xr1, yr1 = x1, y1+h1, x1+w1, y1
        # Coordinates of the next rectangle
        (x2, y2, w2, h2) = cv2.boundingRect(cnts[index+1])
        xl2, yl2, xr2, yr2 = x2, y2+h2, x2+w2, y2

        # Check if current and next overlap
        inter = doOverlap(xl1, yl1 + vertical_pad, xr1, yr1, xl2, yl2 + vertical_pad, xr2, yr2)   
        if inter==True:
            # Update current rectangle
            x1, y1, w1, h1 = merge_overlapping_rectangles(xl1, yl1, xr1, yr1, xl2, yl2, xr2, yr2)
            xl1,yl1, xr1, yr1 = x1, y1+h1, x1+w1, y1

            if index > 0:
                # check with previous rectangle on the list
                x0, y0, w0, h0 = rectangles[len(rectangles) - 1]
                xl0, yl0, xr0, yr0 = x0, y0+h0, x0+w0, y0
                
                inter_previous = doOverlap(xl1, yl1 + vertical_pad, xr1, yr1, xl0, yl0, xr0, yr0)
                if inter_previous == True:
                    # Update current rectangle
                    x1, y1, w1, h1 = merge_overlapping_rectangles(xl0, yl0, xr0, yr0, xl1, yl1, xr1, yr1)
                    # Remove the previous rectangle (x0,y0,w0,h0) from the list 
                    rectangles = rectangles[:-1]  
            # add current rectangle to list
            rectangles.append((x1, y1, w1, h1))

        elif index==0:
            rectangles.append((x1, y1, w1, h1))

        # if current and next don't overlap  
        else:
            # check with previous rectangle on the list
            x0, y0, w0, h0 = rectangles[len(rectangles) - 1]
            xl0, yl0, xr0, yr0 = x0, y0+h0, x0+w0, y0

            inter_previous = doOverlap(xl1, yl1 + vertical_pad, xr1, yr1, xl0, yl0, xr0, yr0)
            if inter_previous == True:
                # Update current rectangle
                x1, y1, w1, h1 = merge_overlapping_rectangles(xl0, yl0, xr0, yr0, xl1, yl1, xr1, yr1)
                # Remove the previous rectangle (x0,y0,w0,h0) from the list 
                rectangles = rectangles[:-1]  
            # add current rectangle to list
            rectangles.append((x1, y1, w1, h1))

            # take care of last contour
            if index == len(cnts) - 2:
                (x1, y1, w1, h1) = cv2.boundingRect(cnts[index + 1])
                rectangles.append((x1, y1, w1, h1))

    return rectangles
        

