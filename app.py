dataset_path = r'D:\plant data\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)'

# Verify the Path =  gives True.
import os
dataset_path = r"D:\plant data\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
print(os.path.exists(dataset_path))


# View the Project Folder Contents
import os
print(os.listdir(dataset_path))

#Load Images with TensorFlow
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    r"D:/plant data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)


#Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

#Training Loader
train_generator = train_datagen.flow_from_directory(
    "D:/plant data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)


#Validation
validation_generator = train_datagen.flow_from_directory(
    "D:/plant data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Building the CNN Model
model = Sequential()

#Layer 1
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D())

#Layer 2
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())

#Layer 3
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D())

#Classifier
model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='softmax'))

#Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20
)

    #Phase 7: Plot Accuracy

    #Plot

    #Training Accuracy
    #Validation Accuracy
    #Training Loss
    #Validation Loss

    #These plots should be included in your project report.


#Evaluate
test_loss,test_accuracy = model.evaluate(test_generator)

print(test_accuracy)    

#lso compute:

#Confusion Matrix
#Classification Report
#Precision
#Recall
#F1-score

#Save Model
model.save("plant_disease_model.keras")

#Prediction  - Load model - Predict uploaded image.
model = tf.keras.models.load_model("plant_disease_model.keras")


#Phase 11: Streamlit App
