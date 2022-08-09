import cv2
import numpy as np
import random as rng

cap = cv2.VideoCapture("./data/sub3test.mp4")
ret, first_frame = cap.read()

def thickness(points):
    #Thickness method, points is an array of points
    min_x = float("inf")
    max_x = -1 * float("inf")

    min_y = float("inf")
    max_y = -1 * float("inf")

    # find max and min of x and y
    for point in points:
        x = point[0][0]
        y = point[0][1]
        # why the double []?
        min_x = min(x, min_x)
        max_x = max(x, max_x)

        min_y = min(y, min_y)
        max_y = max(y, max_y)

    return (max_x - min_x), (max_y - min_y)

def findconvexhull(contourslist):

    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contourslist)):
        hull = cv2.convexHull(contourslist[i])
        hull_list.append(hull)
    # Draw contours + hull results
    #drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contourslist)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(frame, contourslist, i, color)
        cv2.drawContours(frame, hull_list, i, color)
    # Show in a window
    return frame

while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrastgray = 5*gray
    # Different blurs
    img_blur_GB = cv2.GaussianBlur(gray, (3,3), 0) 
    img_blur_BF = cv2.bilateralFilter(gray, 7, 150, 100)
    img_blur_median = cv2.medianBlur(gray, 7)
    img_means = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize = 9, searchWindowSize = 21)
    img_filter1 = cv2.edgePreservingFilter(gray, 2, 25, 1)

    ret, thresh = cv2.threshold(img_blur_BF, 50, 255, cv2.THRESH_BINARY)

    # Displays filtered images
    cv2.imshow('w/o filter', gray)
    cv2.imshow('contrasted', contrastgray)
    cv2.imshow('GB', img_blur_GB)
    cv2.imshow('BF', img_blur_BF)
    cv2.imshow('median', img_blur_median)
    cv2.imshow('thresh', thresh)
    cv2.imshow('meansDN', img_means)
    cv2.imshow('filter1', img_filter1)

    # Shows Canny Edge Detection on filtered images
    edgesNB = cv2.Canny(image=gray, threshold1=50, threshold2=125)
    cv2.imshow('Canny - w/o filter', edgesNB)
    edgesGB = cv2.Canny(image=img_blur_GB, threshold1=50, threshold2=125)
    cv2.imshow('Canny - GB', edgesGB)
    edgesBF = cv2.Canny(image=img_blur_BF, threshold1=50, threshold2=125)
    cv2.imshow('Canny - BF', edgesBF)
    edgesMedian = cv2.Canny(image=img_blur_median, threshold1=25, threshold2=100)
    cv2.imshow('Canny - Median', edgesMedian)

    # detect the contours on the binary image using cv2.ChAIN_APPROX_SIMPLE

    # gets a tuple of contours, each of which is an array of ordered pairs
    contourslist, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #draws contourslist on image_copy1
    image_copy1 = frame.copy()
    cv2.drawContours(image_copy1, contourslist, -1, (255, 0, 0), 2, cv2.LINE_AA)

    #resets biggestcontour, giving it initial value for area comparison
    biggestcontour = contourslist[0]
    # goes through each contour found in the current image, finds biggest contour
    for i in range(len(contourslist)):
        
        currentcnt = contourslist[i]
        # cv2.drawContours(image_copy1, currentcnt, -1, (0, 255, 0), 2, cv2.LINE_AA)

        if cv2.contourArea(currentcnt) > cv2.contourArea(biggestcontour):
            biggestcontour = currentcnt

    #draws and shows biggest contour, thickness, and area
    cv2.drawContours(image_copy1, biggestcontour, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Simple approximation', image_copy1)
    print(thickness(biggestcontour), cv2.contourArea(biggestcontour))

    '''thicknessprod = thickness(biggestcontour)[0]*thickness(biggestcontour)[1]
    squareness = cv2.contourArea(biggestcontour)/thicknessprod
    print(squareness)'''
    
    #hulls
    '''hulls = findconvexhull(contourslist)
    cv2.imshow('Contours', hulls)'''
    
    print("done current image")    

    # breaks
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()