import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import os

os.makedirs("models",exist_ok=True)
img_size=(224,224)
batch_size=32

datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
    )

train_data=datagen.flow_from_directory(
    "dataset/",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'

)
val_data = datagen.flow_from_directory(
    "dataset/",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#Class Weights

classes=np.unique(train_data.classes)
class_weights=compute_class_weight('balanced',classes=classes,y=train_data.classes)
class_weights=dict(enumerate(class_weights))

#Model
base_model= EfficientNetB3(weights='imagenet',include_top=False,input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable=False

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(256,activation='relu')(x)
x=Dropout(0.5)(x)
output=Dense(train_data.num_classes,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=output)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Train

model.fit(train_data, validation_data=val_data,epochs=5,class_weight=class_weights)

#fine-tune


for layer in base_model.layers[-20:]:
    layer.trainable=True

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(train_data,validation_data=val_data,epochs=3)

#Save
model.save("models/skin_disease_model.h5")
