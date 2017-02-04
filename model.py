import ast
from keras.models import Sequential
from keras.layers import Conv2D, Flatten,MaxPooling2D,Dropout,Activation, Dense, ELU
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from preprocessing import *
from keras.preprocessing.image import ImageDataGenerator

# Read train data into array 'logs'
path = 'data_0202/driving_log.csv'
logs = read_data(path)

# Read validation data into array 'logs'
path1 = 'data_0302/driving_log.csv'
logs1= read_data(path1)

# Generator of train data
def image_generator(logs, batch_size=128):         
    while True:          
        X_batch = []
        y_batch = []    
        for i in range(batch_size):  
            
            # Random selection of image            
            idx = np.random.randint(len(logs))   
            data = [i.split('\t', 1)[0] for i in logs[idx]]
            selection = np.random.choice(['centre', 'left', 'right'])

            # Random selection of image            
            # For right and left image add .25 to steering angle            
            if selection == 'centre':            
                _image = cv2.imread(data[0])            
                _angle=ast.literal_eval(data[3])   
                _image_process, _anglenew= format_pic(_image, float(_angle))
                
            if selection == 'left':            
                _image = cv2.imread(data[1])            
                _angle=ast.literal_eval(data[3])
                _image_process, _anglenew= format_pic(_image, (float(_angle)+.25))
            
            if selection == 'right':            
                _image = cv2.imread(data[2])            
                _angle=ast.literal_eval(data[3])
                _image_process, _anglenew= format_pic(_image, (float(_angle)-.25))
                        
            # Remove steering angle '0' randomly to better train CNN           
            if (np.random.uniform()>=.5 and (_anglenew>-.1 and _anglenew<.1)):            
                X_batch.append(_image_process)
                y_batch.append(_anglenew)          
            else: 
                X_batch.append(_image_process)
                y_batch.append(_anglenew)          
            
        X_batch=np.array(X_batch)
        y_batch= np.array(y_batch)        
        X_batch, y_batch= shuffle(X_batch, y_batch)
        
        yield (X_batch, y_batch)        

# Generator of validation data
def image_val_generator(logs1, batch_size=128):         
    while True:          
        X_batch = []
        y_batch = []    
        for i in range(batch_size):  
            
            # Random selection of image (only center data)
            idx = np.random.randint(len(logs1))                            
                        
            data = [i.split('\t', 1)[0] for i in logs1[idx]]

            # Prepare image in the same way as training data
            _image = cv2.imread(data[0])                        
            _image = _image[30:140,50:270]
            _image = cv2.resize(_image, (200,66))
            _image = cv2.cvtColor(_image,cv2.COLOR_RGB2HSV)
        
            _angle= ast.literal_eval(data[3])       

            # Flipping data in order to have a good distriution of all angles
            if np.random.uniform()>.5:
                _image=cv2.flip(_image,1)
                if _angle!=0:
                    _angle = -_angle
                    
            X_batch.append(_image)
            y_batch.append(_angle)      
                        
        X_batch=np.array(X_batch)
        y_batch= np.array(y_batch)
        X_batch, y_batch= shuffle(X_batch, y_batch)

        yield (X_batch, y_batch)  
        
# Implementing the Nvidia CNN - outputs the steering angle
# Drop-out after each CNN layer in order to reduce overfitting
model = Sequential()
model.add(Conv2D(24, 5, 5, subsample=(2,2), input_shape=(66,200,3,),  border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.3))
model.add(Conv2D(36, 5, 5, subsample=(2,2),  border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.3))
model.add(Conv2D(48, 5, 5, subsample=(2,2),  border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.3))
model.add(Conv2D(64, 3, 3,  border_mode='valid'))
model.add(ELU())
model.add(Dropout(0.3))
model.add(Conv2D(64, 3, 3,  border_mode='valid'))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(ELU())
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))
    
# Saving of best model by validation mean squared error
check_point = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
# Stop training when there is no improvment. 
stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

# Compilation of model with Adam optimizer 
model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
    
# Saving model as JSON
model_json = model.to_json()
with open("model.json", "w") as model_file:
    model_file.write(model_json)

# Run model
history = model.fit_generator(image_generator(logs), samples_per_epoch=190*128, nb_epoch=10, verbose=1,callbacks=[check_point,stopping], validation_data=image_val_generator(logs1),nb_val_samples=50*128)
