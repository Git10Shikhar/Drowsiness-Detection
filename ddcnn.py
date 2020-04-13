from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D


def generator(dir):
    
     generator=image.ImageDataGenerator(rescale=1./255)
     
     generator.flow_from_directory(dir,batch_size=32,shuffle=True,color_mode='grayscale',
                                   class_mode='categorical',target_size=(24,24))
     
train_batch= generator('data/train')
valid_batch= generator('data/valid')

SPE= len(train_batch.classes)//32  # steps per epoch

VS = len(valid_batch.classes)//32  # validation steps


model = Sequential ([
    
    Conv2D(32, (3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(1,1
                 ),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(1,1),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(1,1),

    Dropout(0.25),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    
    Dropout(0.5),
    
    Dense(2, activation='softmax')
    
    ])
     

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)


model.save('cnn.h5', overwrite=True)