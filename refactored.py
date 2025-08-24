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

def perspective_transform(img):
    if len(click_positions) == 4:
        width = 640
        height = 480
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv.getPerspectiveTransform(np.array(click_positions, dtype=np.float32), dst)
        img = cv.warpPerspective(img, M, (width, height))

    return img


def detect_lines(img):
    cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cimg = cv.medianBlur(cimg, 5)

    edges = cv.Canny(cimg, 50, 150, apertureSize=3)
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

    return img
    #cv.imshow('main window', img)

def detect_circles(img):
    cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cimg = cv.medianBlur(cimg, 5)
    #cimg = cv.cvtColor(cimg, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(
        cimg,
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
                result = cv.pointPolygonTest(pts, (int(i[0]), int(i[1])), False)
                if result > 0:
                    cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            cv.putText(img, "Click corners of table in clockwise order", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv.putText(img, "No circles detected.", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    img = detect_lines(img)

    img = perspective_transform(img)

    return img
    #cv.imshow('main window', cimg)
    

def main():
    cv.namedWindow('main window')
    cv.setMouseCallback('main window', mouseInput)

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        
        img = detect_circles(frame)
        cv.imshow('main window', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()

    #detect_circles()
    #detect_circles_from_camera()

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()