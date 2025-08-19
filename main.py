import numpy as np
import cv2 as cv

def detect_circles():
    img = cv.imread('testImage.jpg', cv.IMREAD_GRAYSCALE)
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(
        img,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=50,
        param2=20,
        minRadius=10,
        maxRadius=16
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        cv.putText(cimg, "No circles detected.", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv.imshow('detected circles', cimg)
    cv.waitKey(0)

def detect_circles_from_camera():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img = cv.medianBlur(img, 5)
        cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        circles = cv.HoughCircles(
            img,
            cv.HOUGH_GRADIENT,
            dp=1.2,
            minDist=10,
            param1=50,
            param2=20,
            minRadius=10,
            maxRadius=16
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            cv.putText(cimg, "No circles detected.", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv.imshow('detected circles', cimg)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def main():
    detect_circles()
    #detect_circles_from_camera()

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()