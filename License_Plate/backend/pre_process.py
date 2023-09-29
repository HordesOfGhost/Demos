import numpy as np
from PIL import Image
import io
import cv2

# convert image to numpy

def image_to_numpy(image_data):
    if (image_data) != 0:
        image_array= Image.open(io.BytesIO(image_data)).convert('RGB')
        cv2_image=cv2.cvtColor(np.array(image_array), cv2.COLOR_BGR2RGB)
        cv2_image = cv2.resize(cv2_image, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
        return cv2_image
    return None