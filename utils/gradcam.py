# utils/gradcam.py
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf

def get_cpu_safe_gradcam(model, img_array, last_conv_layer_name=None):
    """
    CPU-safe Grad-CAM (no backprop) for single image.
    """
    # Step 1: Detect last Conv2D layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and 'conv2d' in str(layer.__class__).lower():
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No Conv2D layer found in model.")

    conv_layer = model.get_layer(last_conv_layer_name)

    # Step 2: Create a model to output conv layer + predictions
    cam_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    conv_output, predictions = cam_model(img_array)
    conv_output = conv_output[0].numpy()  # remove batch dim
    pred_idx = np.argmax(predictions[0])

    # Step 3: Get weights from the last Dense layer
    final_dense = model.layers[-1]  # assumes last layer is Dense
    weights = final_dense.get_weights()[0]  # shape: (num_features, num_classes)
    class_weights = weights[:, pred_idx]  # weights for predicted class

    # Step 4: Compute weighted sum of conv feature maps
    cam = np.zeros(conv_output.shape[:2], dtype=np.float32)
    for i, w in enumerate(class_weights):
        if i < conv_output.shape[-1]:  # safety check
            cam += w * conv_output[:, :, i]

    cam = np.maximum(cam, 0)
    if np.max(cam) != 0:
        cam /= np.max(cam)

    # Boost visibility
    cam = np.clip(cam * 2, 0, 1)

    # Step 5: Resize heatmap to original image size
    heatmap = Image.fromarray(np.uint8(cam*255)).resize((img_array.shape[2], img_array.shape[1]))
    heatmap = np.array(heatmap)/255.0

    # Step 6: Apply jet colormap
    jet_heatmap = cm.jet(heatmap)[:, :, :3]

    # Step 7: Superimpose on original image
    original_img = np.uint8(img_array[0]*255)
    alpha = 0.6
    superimposed_img = jet_heatmap*255*alpha + original_img*(1-alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = Image.fromarray(superimposed_img)

    return superimposed_img