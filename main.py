import cv2
import torch
import numpy as np

path = 'C:/Users/Yashashvi Mann/Desktop/metro/best_4.pt'

model = torch.hub.load(r'C:/Users/Yashashvi Mann\Desktop/metro/yolov5', 'custom',path, force_reload=True, source='local')

cap=cv2.VideoCapture("metro.mp4")
if (cap.isOpened() == False): 
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# size = (frame_width, frame_height)

# # Below VideoWriter object will create
# # a frame of above defined The output 
# # is stored in 'filename.avi' file.
# result = cv2.VideoWriter('metro.avi', 
#                         cv2.VideoWriter_fourcc(*'XVID'),
#                         10, size)
# count = 0

def Cordinates(event , x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(frame,(x,y),4,(0,255,0),-1)
        strXY = '('+str(x)+','+str(y)+')'
        cv2.putText(frame,strXY,(x+10,y-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),1)
        cv2.imshow("FRAME",frame)

a1 = [(140,119),(264,143),(352,169),(356,271),(313,304),(242,321),(175,230)]
a2 = [(714,170),(882,158),(842,303),(653,291)]

while True:
    ret,frame=cap.read()
    if not ret:
        break
    # result.write(frame)

    frame=cv2.resize(frame,(1020,500))
    results = model(frame)
    frame = np.squeeze(results.render())
    
    a = results.pandas().xyxy[0]
    b = a.index.stop
    #print(b)
    c = int((b/240)*100)
    cv2.putText(frame,"Space Occupied - "+str(int(c))+"%",(550,590),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow("FRAME",frame)
    cv2.setMouseCallback('FRAME',Cordinates) 
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
# result.release()
cv2.destroyAllWindows()