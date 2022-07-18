#editor: Williams Bobadilla
#created_at: 09 may 2022
#edited_by: Williams Bobadilla
#edited_at: 09 may 2022
#description: example of yolov3, adapted with 
#version: 1.0.0



import traceback
import cv2
import numpy as np


CONFIDENCE_THRESH = 0.7
NMS_THRESH = 0.3
whT = 320 # for the net 

VIDEO_PATH = 'vehicle.mp4' #"cow1.jpg"


classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print("classNames", classNames)
#model config
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)



def resize(img, percent = 60):
    """
    Resize an img in base of a percent given.
        Params:
            img(nd.Array): image to be resized
            percent(int):percent to resize from the original size
        Returns: 
            resized(nd.Array): image resized
    """
    scale_percent = percent # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def find_objects(outputs, img, draw = True):
    """
    Function to find object in the img frame passed
        Params: 
            outputs (tuple<nd.Array>): tuple of arrays that contains the output from  net.forward() method.
            img (nd.Array): array of the img.
            draw (bool): bool to indicates to draw over the img or not.
        Returns: 
            result(dic): dic of the results, see in notes for more information about the structure.
        Notes: 
            result = {
                "bbox": [[x,y,w,h],[x,y,w,h],...]
                "classIds": [0,0,...],
                "confs": [5.2,6.3....]
            }
    """
    hT,wT, cT = img.shape
    print(f"w: {wT}, h: {hT}")
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:] # ? 
            classId = np.argmax(scores) 
            #print("classId", classId)
            confidence = scores[classId]
            if confidence > CONFIDENCE_THRESH:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y =int( det[0]*wT - w/2), int(det[1]*hT - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indexes = cv2.dnn.NMSBoxes(bbox, confs, CONFIDENCE_THRESH, NMS_THRESH)
    for i in indexes:
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]]} {round(confs[i]*100,2)}%',(x,y+10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
        #print(f'{classNames[classIds[i]]} {confs[i]*100}%')
    result = {
        "bbox": bbox,
        "classIds": classIds,
        "congfs": confs
    }
    return result



# ---------------------------MAIN-------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    try:
        success, img = cap.read()
        
        #convert img input to blob for the net
        #img = resize(img,60)
        whT= whT 
        blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1, crop=False)
        # set the input of the net
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)
        
        find_objects(outputs, img)

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        break

print("cerrando")
cv2.destroyAllWindows()
cap.release()
