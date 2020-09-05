import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread("photo.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)  #minNeighbors ideal value 5 to check how many near neighbour.

for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)  #(start cordinate),(end cordinate),(color of line),(width of rectangle)

resized=cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
cv2.imshow("Gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()