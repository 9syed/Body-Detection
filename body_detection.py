import cv2

body_casecade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
video = cv2.VideoCapture('video2.mp4')

while True:
    vid, color_image = video.read()

    if vid == False:
        break

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    body = body_casecade.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in body:
        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 255),3)
        cv2.imshow("Body Detector", color_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()