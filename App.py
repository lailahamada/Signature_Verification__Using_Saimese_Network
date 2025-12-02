# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import gradio as gr
from PIL import Image
import os
import sys

# Image size used during model training (constant)
IMG_SIZE = 224

# Verification threshold (55%): Defines the minimum required similarity for a match
VERIFICATION_THRESHOLD = 0.55

# 1. Custom Layer Definitions (Mandatory for model loading)
# These definitions are necessary because the trained model uses these custom layers.
class L2Normalization(layers.Layer):
    """Applies L2 normalization to the input vector."""
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)

class L1Distance(layers.Layer):
    """Calculates the absolute difference between two vectors."""
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

# Required list for loading the model with custom layers
CUSTOM_OBJECTS = {
    "L2Normalization": L2Normalization,
    "L1Distance": L1Distance
}

# 2. Model Path Definition
MODEL_PATH = '/kaggle/working/Siamese_signature.keras'

# 3. Fallback Model Building Function (Used if the trained file is not found)
def build_fallback_model():
    """Builds a simple Siamese model with random weights."""
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input_image")
    # Simple CNN backbone structure
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    embed = layers.Dense(256, activation='relu')(x)
    embed = L2Normalization()(embed)
    
    input_2 = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image_2")
    
    # Create the backbone model for re-use
    backbone_model = Model(inputs, embed)
    
    embed_1 = backbone_model(inputs)
    embed_2 = backbone_model(input_2)
    
    distance = L1Distance()([embed_1, embed_2])
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = Model(inputs=[inputs, input_2], outputs=output, name="SiameseFallback")
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# 4. Model Loading
try:
    # Attempt to load the trained model
    siamese_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}. Ready for prediction.")
except Exception as e:
    # Fallback plan if the file is not found
    print(f"❌ Failed to load model from {MODEL_PATH}. Please ensure the saved file is in the notebook path.")
    print(f"Building a fallback model with random weights instead of stopping. Error: {e}")
    siamese_model = build_fallback_model()
    # Exit to prevent use of the untrained model if the trained one is mandatory
    sys.exit("Stopped: Trained model not found. Please ensure the file path is correct.")


# 5. Image Pre-processing Function
def preprocess_image(img_arr):
    """
    Processes the image: converts to grayscale, resizes, and normalizes to fit the model input.
    """
    if img_arr is None:
        return None
    
    # Convert to grayscale if not already
    if img_arr.ndim == 3 and img_arr.shape[-1] == 3:
        img = Image.fromarray(img_arr).convert('L')
        img_arr = np.array(img)
    elif img_arr.ndim == 2:
        pass
    else:
        try:
            img = Image.fromarray(img_arr).convert('L')
            img_arr = np.array(img)
        except:
            return None
        
    # Resize
    img = tf.image.resize(img_arr[..., np.newaxis], [IMG_SIZE, IMG_SIZE])
    
    # Normalization
    img = tf.cast(img, tf.float32) / 255.0
    
    # Add batch dimension
    return tf.expand_dims(img, axis=0)

# 6. Gradio Prediction Function
def predict_match(image1, image2):
    """
    Takes two images, processes them, and uses the model for prediction.
    """
    img1_processed = preprocess_image(image1)
    img2_processed = preprocess_image(image2)

    if img1_processed is None or img2_processed is None:
        # Error message for the user in English
        return gr.Label("Please upload a valid image for both inputs for comparison.", color="yellow", label="Verification Result")

    # Perform prediction using the trained model
    try:
        # Prediction requires a list of inputs: [image1, image2]
        prediction = siamese_model.predict([img1_processed, img2_processed], verbose=0)[0][0]
    except Exception as e:
        # Error message for the user in English
        return gr.Label(f"An error occurred during prediction. Error: {e}", color="red", label="Prediction Error")
    
    # Interpret the result and compare against the threshold (0.55)
    similarity_percentage = prediction * 100
    
    # Check for match based on the threshold
    if prediction >= VERIFICATION_THRESHOLD:
        match_status = "Genuine ✅ (Match)"
        color = "green"
    else:
        match_status = "Forged ❌ (No Match)"
        color = "red"
        
    output_text = match_status
    
    return gr.Label(output_text, color=color, label="Verification Result")

# 7. Create Gradio Interface
iface = gr.Interface(
    fn=predict_match,
    inputs=[
        # Input labels in English
        gr.Image(label="(Image 1)", type="numpy"),
        gr.Image(label="(Image 2)", type="numpy")
    ],
    # Output label in English
    outputs=gr.Label(label="Verification Result"),
    # Title and description in English
    title="Signature Verification ✍️🖋️",
    description=f"Upload two signatures to classify as Genuine ✅ or Forged ❌.",
    allow_flagging="never",
    # Interface theme
    theme=gr.themes.Soft()
)

# 8. Launch the Application
iface.launch(share=True)
