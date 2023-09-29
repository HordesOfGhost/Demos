from flask import Flask,request,flash,jsonify,render_template
import numpy as np
import cv2,os,atexit
from backend.pre_process import *

import torch
from ultralytics import YOLO

# Select Device
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# Initialize vehicle_detection Models
path = 'yolov7/yolov7-e6e.pt'
yolo_model_v7 = torch.hub.load("WongKinYiu/yolov7","custom",f"{path}",trust_repo=True,verbose=False)
yolo_model_v7.to(device)
yolo_model_v8 = YOLO("models/yolov8x.pt")
yolo_model_v8.to(device)

app=Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.rout('/upload/')
def upload_image():
    if request.method == 'POST':
        
        image_path = 'static/images/uploaded.jpg'
        image = request.files['image']        

        if image:
            image_data = image.read()
            image_numpy = image_to_numpy(image_data)
            cv2.imwrite(image_path,image_numpy)
            return render_template('license_plate_detection.html',prediction=prediction,recommended_song = recommended_song)
        
    return render_template('image_detection.html')


# On exit delete all images

def OnExitApp():
    try:
        os.remove('static/images/uploaded.jpg')
    except:
        pass

atexit.register(OnExitApp)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5050)