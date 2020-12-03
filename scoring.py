import json
from ibm_watson import VisualRecognitionV4
from ibm_watson.visual_recognition_v4 import FileWithMetadata, AnalyzeEnums
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from matplotlib import pyplot as plt
import cv2
import uuid
import os
import time
import imutils

# image capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

api_key = "eKioDSuJ-NBnZZfwUTBPEMToM5KylgGJOVEzkrgPlCwJ"
URL = "https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/5b21ee76-dd63-4677-b83e-04b9021f08ce"
collection_id = "93d9fe7d-d7da-4c88-b40b-f49c56014f14"

authenticator = IAMAuthenticator(api_key)
service = VisualRecognitionV4('2018-03-19', authenticator=authenticator)
service.set_service_url(URL)

path = ''
while True:
    # captures form webcam
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=750)
    imgname = './Images/Temp/{}.jpg'.format(str(uuid.uuid1()))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # saves image taken
        cv2.imwrite(imgname, frame)
        path = imgname
        break

print(path)
# path = './Images/Mask/1bbca78c-3516-11eb-bccb-06ab6e8ff6f9.jpg'
# print(path)
with open(path, 'rb') as mask_img:
    analyze_images = service.analyze(collection_ids=[collection_id],
                                     features=[
                                         AnalyzeEnums.Features.OBJECTS.value],
                                     images_file=[FileWithMetadata(mask_img)]).get_result()
print(analyze_images)
# Visualise
obj = analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['object']
coords = analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['location']

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# creates rec
if obj == "Mask":
    img = cv2.rectangle(img, (coords['left'], coords['top']), (coords['left'] +
                                                               coords['width'], coords['top']+coords['height']), (0, 255, 0), 10)

if obj == "NoMask":
    img = cv2.rectangle(img, (coords['left'], coords['top']), (coords['left'] +
                                                               coords['width'], coords['top']+coords['height']), (255, 0, 0), 10)
# font = cv2.FONT_HERSHEY_SIMPLEX
# img = cv2.putText(img, text=obj, org=(coords['left'], coords['top'], coords['left'] +
#                                       coords['width'], coords['top']+coords['height']), fontFace=font, fontScale=2, color=(0, 255, 0) thinckness=5, lineType=cv2.LINE_AA)

plt.imshow(img)
plt.show()
