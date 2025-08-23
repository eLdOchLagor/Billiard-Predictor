import numpy as np
import cv2 as cv

click_positions = []

def mouseInput(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(click_positions) < 4:
            click_positions.append((x, y))
            print(f"Mouse clicked at: ({x}, {y})")
            print(f"Stored positions: {click_positions}")
        else:
            print("Already stored 4 positions.")

def detect_circles(img):
    #img = cv.imread('testImage.jpg', cv.IMREAD_GRAYSCALE)
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

    #cv.imshow('main window', cimg)
    detect_lines(cimg)
    cv.waitKey(0)

def detect_circles_from_camera(img):
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
            if len(click_positions) == 4:
                pts = np.array(click_positions, np.int32)
                for i in circles[0, :]:
                    # Check if the circle center is inside the polygon
                    result = cv.pointPolygonTest(pts, (int(i[0]), int(i[1])), False)
                    if result > 0:
                        # draw the outer circle
                        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        # draw the center of the circle
                        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
                
        else:
            cv.putText(cimg, "No circles detected.", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        #cv.imshow('detected circles', cimg)
        detect_lines(cimg)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def detect_lines(img):
    edges = cv.Canny(img, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(click_positions) == 4:
        pts = np.array(click_positions, np.int32)
        pts_poly = pts.reshape((-1, 1, 2))
        #cv.fillPoly(img, [pts_poly], (0, 0, 255))
        cv.polylines(img, [pts_poly], True, (0, 0, 255), 2)
    cv.imshow('main window', img)

def main():
    cv.namedWindow('main window')
    cv.setMouseCallback('main window', mouseInput)

    #detect_circles()
    detect_circles_from_camera()

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()