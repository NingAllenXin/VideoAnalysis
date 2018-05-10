import imutils
import cv2

body_cascade = cv2.CascadeClassifier('/Users/Allen/Desktop/final/haarcascades/haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('/Users/Allen/Desktop/final/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('/Users/Allen/Desktop/final/haarcascades/haarcascade_eye.xml')
Logo_height_inreal = 28

def detect_logo(img):
    template = cv2.imread('temp1.jpeg', 0)
    # bh, bw = img.shape[::-1]
    # template = imutils.resize(template, width=int(bw * 0.5))
    h, w = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    threshold = 0.6
    if min_val > threshold:
        checked = False
    else:
        checked = True

    return checked, top_left, bottom_right, h


cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    img = imutils.resize(img, width=1000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply template Matching
    body = body_cascade.detectMultiScale(gray, 1.05, 3)
    for (bx, by, bw, bh) in body:
        bh = bh+40
        b_gray = gray[by:by + bh, bx:bx + bw]
        b_color = img[by:by + bh, bx:bx + bw]
        if bw > 100 and bh > 100:
            isLogo, topLeft, bottomRight, logo_h = detect_logo(b_gray)
            if isLogo:
                #draw body
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
                #draw logo
                cv2.rectangle(b_color, topLeft, bottomRight, (0, 255, 255), 2)

                #calculate height
                ratio = bh / logo_h
                height = ratio * Logo_height_inreal * 1.3
                #if (height > 150):
                print "Human height : {0}cm".format(height)

                faces = face_cascade.detectMultiScale(b_gray, 1.2, 3)
                for (x,y,w,h) in faces:
                    cv2.rectangle(b_color,(x,y),(x+w,y+h),(0,255,0),3)
                    roi_gray = b_gray[y:y+h, x:x+w]
                    roi_color = b_color[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 3)
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
