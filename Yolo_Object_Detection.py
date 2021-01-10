import cv2
import numpy as np

# load the pre-trained model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg.txt")
classes = []

# Make a list of labels
with open("coco.names.txt", 'r') as names:
     classes = [line.strip() for line in names.readlines()]

# Get layer names in the model
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, (len(classes), 3))        # To use different colors for each label

def yolo_detection(img):
     height, width, channels = img.shape
     blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, mean=(0, 0, 0), size=(416, 416), swapRB=True)
     net.setInput(blob)        # set the input blob to model
     outputs = net.forward(output_layers)
     classIds = []
     confidences = []
     boundingBoxes = []
     for out in outputs:
          for detection in out:
               scores = detection[5 : ]
               class_id = np.argmax(scores)
               confidence = scores[np.argmax(scores)]
               if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    classIds.append(class_id)
                    boundingBoxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
     # To remove the overlapped bounding boxes
     indexes = cv2.dnn.NMSBoxes(boundingBoxes, confidences, 0.5, 0.4)
     if len(indexes):
          indexes = indexes.reshape((-1))
     for i in indexes:
          x, y, w, h = boundingBoxes[i]
          name = classes[classIds[i]]
          color = colors[classIds[i]]
          cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
          cv2.putText(img, name, (x, y + 100), cv2.FONT_HERSHEY_COMPLEX, 1, color, 3)

# To detect objects in video or real-time webcam
def Video(path):
     try:
          webcam = cv2.VideoCapture(path)
          frameWidth = 1280
          frameHeight = 720
          webcam.set(3, frameWidth)
          webcam.set(4, frameHeight)
          webcam.set(10, 250)
          while True:
               success, img = webcam.read()
               yolo_detection(img)
               cv2.imshow("img", img)
               if cv2.waitKey(30) & 0XFF == ord('q'):
                    break
          webcam.release()
     except:
          print("Invalid path")
          return

# To detect objects in Image
def Image(path):
     try:
          img = cv2.imread(path)
          yolo_detection(img)
          cv2.imshow("Image", img)
          cv2.waitKey(0)
     except:
          print("Invalid path")
          return

if __name__ == "__main__":
     print("You can choose from the following options \n\t 1. Webcam \n\t 2. Video \n\t 3. Image")
     choice = input("What is your option: ")
     path = 0

     if choice == '1' or choice.lower() == "webcam":
          print("Press 'q' to quit")
          Video(path)
     elif choice == '2' or choice.lower() == "video":
          path = input("Please specify the path: ")
          print("Press 'q' to quit")
          Video(path)
     elif choice == '3' or choice.lower() == "image":
          path = input("Please specify the path: ")
          Image(path)
     else:
          print("Sorry! you have chosen invalid option")

     print("Bye! Have a nice day!")
     cv2.destroyAllWindows()

