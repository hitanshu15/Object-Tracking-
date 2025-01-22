# Object-Tracking-
import cv2 as cv 
import supervision as sv
from ultralytics import YOLO

yolo10n = YOLO("yolov10n")

model = YOLO("best.pt")
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

camera = cv.VideoCapture("C:/Users/HP/Videos/WhatsApp Video 2025-01-14 at 22.03.47_0c1b7e21.mp4")# enter path of the video 
camera.set (3,1000) # weight 
camera.set(4,1000) # height 
camera.set(10,300) # brightness 

if not camera.isOpened():
    print("Enble to load open camer feed")


while True:
    success, webcam = camera.read()
    if not success:
        break

    results = model(webcam)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(scene=webcam, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)


    cv.imshow("webcam", annotated_image)

    if cv.waitKey(1)& 0xFF == ord("0"):
        break

camera.release()
cv.destroyAllWindows()
