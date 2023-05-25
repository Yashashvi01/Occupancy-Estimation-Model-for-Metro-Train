import cv2
import torch
import numpy as np

path = 'C:/Users/Yashashvi Mann/Desktop/metro/best_4.pt'

model = torch.hub.load(r'C:/Users/Yashashvi Mann\Desktop/metro/yolov5', 'custom',path, force_reload=True, source='local')

cap=cv2.VideoCapture("metro.mp4")
if (cap.isOpened() == False): 
    print("Error reading video file")

# Define the region of interest
a1 = [(142,140),(190,333),(359,163),(378,318)]
# a2 = [(714,170),(882,158),(842,303),(653,291)]
# roi = np.array([a1,a2])

# Function to check if a point lies within a polygon
def point_in_poly(x, y, poly):
    if len(poly) == 0:
        return False
    poly = np.array(poly, dtype=np.int32)
    if poly.ndim == 3:
        poly = poly.squeeze()
    if poly.dtype != np.int32:
        poly = poly.astype(np.int32)
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0


while True:
    ret,frame=cap.read()
    if not ret:
        break

    # Resize the frame
    frame=cv2.resize(frame,(1020,500))

    # Get the detection results and render them on the frame
    results = model(frame)
    frame = np.squeeze(results.render())

    # Count the number of detection boxes that lie within the ROI
    count = 0
    for box in results.pandas().xyxy[0].values:
        # Calculate the center coordinates of the box
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        
        # Check if the center coordinates lie within the ROI
        if point_in_poly(center_x, center_y, a1):
            count += 1
    
    
    # Display the count on the frame
    cv2.putText(frame, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("FRAME",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
