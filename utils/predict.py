import numpy as np

def preprocess_image(image):
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(model, img_array, class_names):
    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds)
    return class_names[class_index], confidence