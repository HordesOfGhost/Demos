from flask import Flask,request,flash,jsonify,render_template
import numpy as np
import cv2,os,atexit
import shutil
from backend.image_functions import *
from backend.detect import *
import torch
from ultralytics import YOLO

# Select Device
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# Initialize license_plate_detection Models
license_detection_model = YOLO('models/license_15k.pt')
license_detection_model.to(device)

# Initialize vehicle_detection Models
path = 'models/yolov7/yolov7-e6e.pt'
yolo_model_v7 = torch.hub.load("WongKinYiu/yolov7","custom",f"{path}",trust_repo=True,verbose=False)
yolo_model_v7.to(device)
yolo_model_v8 = YOLO("models/yolov8x.pt")
yolo_model_v8.to(device)

app=Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/upload/', methods = ['GET','POST'])
def upload_image():
    if request.method == 'POST':
        
        image_path = 'static/images/uploaded.jpg'
        image = request.files['image']        

        if image:
            image_data = image.read()
            image_numpy = image_to_numpy(image_data)
            cv2.imwrite(image_path,image_numpy)
            final_blurred_image, vehicle_detected , vehicle_blurred_license, information = BlurFaces(image = image_numpy, yolo_model_v7 = yolo_model_v7, yolo_model_v8 = yolo_model_v8, license_detection_model = license_detection_model).blur_license_plates()
            
            cv2.imwrite('static/images/blurred.jpg',final_blurred_image)
            
            if os.path.exists('static/images/vehicles_roi'):
                shutil.rmtree('static/images/vehicles_roi')
                shutil.rmtree('static/images/blurred_license_plate_vehicles_roi')
            
            shutil.os.makedirs('static/images/blurred_license_plate_vehicles_roi', exist_ok=True)
            shutil.os.makedirs('static/images/vehicles_roi', exist_ok=True)
            
            if vehicle_detected != None:
                vehicles_roi_files = save_images('static/images/vehicles_roi',vehicle_detected)
                blurred_license_plate_vehicles_roi_files = save_images('static/images/blurred_license_plate_vehicles_roi',vehicle_blurred_license)

                return render_template('license_plate_detection.html',information = information, vehicles_roi_files = vehicles_roi_files , blurred_license_plate_vehicles_roi_files = blurred_license_plate_vehicles_roi_files)
            else:
                return render_template('license_plate_not_detection.html',information = information)

    return render_template('license_plate_detection.html',information = information)


# On exit delete all images

# def OnExitApp():
#     try:
#         if os.path.exists('static/images/vehicles_roi'):
#             shutil.rmtree('static/images/vehicles_roi')
#             shutil.rmtree('static/images/blurred_license_plate_vehicles_roi')
#     except:
#         pass

# atexit.register(OnExitApp)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5050,debug=True)