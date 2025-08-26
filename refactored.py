import numpy as np
import cv2 as cv

click_positions = []
global lines
global circles
global cue_position
global direction

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
        width = 1000
        height = 500
        #height, width = img.shape[:2]
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv.getPerspectiveTransform(np.array(click_positions, dtype=np.float32), dst)
        img = cv.warpPerspective(img, M, (width, height))

    return img


def detect_lines(img):
    cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cimg = cv.medianBlur(cimg, 5)

    edges = cv.Canny(cimg, 50, 150, apertureSize=3)

    lines = None
    line_positions = []

    if len(click_positions) == 4:
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            line_positions.append((mid_x, mid_y))

        if line_positions:
            avg_x = sum(pos[0] for pos in line_positions) / len(line_positions)
            avg_y = sum(pos[1] for pos in line_positions) / len(line_positions)
            cue_position = np.array([avg_x, avg_y], dtype=np.int32)
            cv.circle(img, (cue_position[0], cue_position[1]), 2, (255, 0, 0), 3)

            x1, y1, x2, y2 = line[0]
            direction = np.array([x2-x1, y2-y1])
            startPoint = (cue_position + direction * 5).astype(int)
            endPoint = (cue_position - direction * 5).astype(int)
            cv.line(img, tuple(startPoint), tuple(endPoint), (255, 0, 0), 2)
            #print(direction)

    if len(click_positions) == 4:
        pts = np.array(click_positions, np.int32)
        pts_poly = pts.reshape((-1, 1, 2))
        #cv.fillPoly(img, [pts_poly], (0, 0, 255))
        cv.polylines(img, [pts_poly], True, (0, 0, 255), 2)

    return img

def detect_circles(img):
    cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cimg = cv.medianBlur(cimg, 5)

    circles = None

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
                cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            cv.putText(img, "Click corners of table in clockwise order", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv.putText(img, "No circles detected.", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return img
    #cv.imshow('main window', cimg)
    

def main():
    cv.namedWindow('main window')
    cv.setMouseCallback('main window', mouseInput)

    cap = cv.VideoCapture(0)

    while True:
        #ret, frame = cap.read()
        #if not ret:
            #break

        #img = frame
        
        img = cv.imread('testImage.jpg')

        img = perspective_transform(img)
        img = detect_circles(img)
        img = detect_lines(img)
        

        cv.imshow('main window', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()