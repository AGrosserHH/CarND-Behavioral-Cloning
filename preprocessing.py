import csv, cv2
import numpy as np

# Read data file in array
def read_data(path):
    _logs=[]
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            _logs.append(row)
    return _logs

# Tranlate image by max. 50 pixels 
def trans_image(_image,steer,trans_range):    
    rows, cols, chan = _image.shape

    # Horizontal translation and 0.008 steering compensation per pixel    
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*.4    
    
    Trans_M = np.float32([[1,0,tr_x],[0,1,0]])
    image_tr = cv2.warpAffine(_image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

# Crop image 
def crop_image(_image, y1, y2, x1, x2):
    return _image[y1:y2, x1:x2]

# Image Processing 
def format_pic(_image, _angle):        
    shape_y = _image.shape[0]
    shape_x = _image.shape[1]
    
    # Normalize image to HSV 
    _image=cv2.cvtColor(_image,cv2.COLOR_RGB2HSV)
        
    # Translate image     
    trans_range = 50
    _image,  _angle = trans_image(_image, _angle, trans_range)
    
    # Crop image   
    _image = crop_image(_image, 20, 140, 0+trans_range, shape_x-trans_range)
    
    # Resizing
    _res = cv2.resize(_image,(200,66))             
    
    # Flip image randomly
    _angle_cor = _angle     
    
    if np.random.uniform()>.5:
        _res=cv2.flip(_res,1)
        if _angle_cor!=0:
            _angle_cor = -_angle            
            
    return _res, np.float32(_angle_cor)
