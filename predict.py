import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras  # FIX: Added to support legacy Keras 2 layer loading in modern TF envs
from PIL import Image

# 1. Image Preprocessing Utility
def process_image(image_np):
    """Converts a raw numpy image array into a normalized 224x224 tensor array."""
    image = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()

# 2. Core Prediction Logic
def predict(image_path, model, top_k):
    """Predicts the top K classes for an image using the loaded Keras model."""
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_image = process_image(test_image)
    
    # Expand dimensions to fit batch format: (1, 224, 224, 3)
    input_batch = np.expand_dims(processed_image, axis=0)
    
    # Run Inference
    predictions = model.predict(input_batch, verbose=0)
    
    # Extract top K probabilities and indices
    probs, indices = tf.math.top_k(predictions, k=top_k)
    
    top_probs = probs.numpy()[0]
    top_classes = [str(idx) for idx in indices.numpy()[0]]
    
    return top_probs, top_classes

# 3. Main Command Line Interface Execution
def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Predict flower species from an image using a trained Keras model.")
    
    # Positional Arguments
    parser.add_argument("image_path", type=str, help="Path to the input flower image.")
    parser.add_argument("model_path", type=str, help="Path to the saved HDF5 Keras model (.h5).")
    
    # Optional Arguments
    parser.add_argument("--top_k", type=int, default=5, help="Return the top K most likely classes.")
    parser.add_argument("--category_names", type=str, default=None, help="Path to a JSON file mapping labels to flower names.")
    
    args = parser.parse_args()
    
    # Load the trained Keras model with its custom TF Hub dependencies
    print(f"Loading model from '{args.model_path}'... Please wait.")
    try:
        # FIX: Swapped tf.keras for tf_keras to handle Keras 2/3 model compatibility safely
        model = tf_keras.models.load_model(
            args.model_path, 
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run the prediction pipeline
    print("Running inference...")
    probs, classes = predict(args.image_path, model, args.top_k)
    
    # Map class integers to real names if JSON mapping file is provided
    if args.category_names:
        try:
            with open(args.category_names, 'r') as f:
                class_names = json.load(f)
            display_labels = [class_names.get(c, f"Unknown Class {c}").title() for c in classes]
        except Exception as e:
            print(f"Warning: Could not read category map file. Falling back to numeric IDs. Error: {e}")
            display_labels = classes
    else:
        display_labels = classes

    # Print out results cleanly to the terminal console
    print("\n=== PREDICTION RESULTS ===")
    for i in range(len(probs)):
        print(f"Rank {i+1}: {display_labels[i]:<30} | Probability: {probs[i]:.4%}")

if __name__ == "__main__":
    main()