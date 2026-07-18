from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = r"D:\plant data\New Plant Diseases Dataset(Augmented)\train"
valid_path = r"D:\plant data\New Plant Diseases Dataset(Augmented)\valid"

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print(train_generator.class_indices)