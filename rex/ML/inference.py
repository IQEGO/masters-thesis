from PyRexExt import REX  # important, don't remove!
import os
import cv2
import numpy as np
import time
import csv
from pathlib import Path
import onnxruntime as ort
import sample_data  # Import our sample data module

inference_interface = None  # global variable to keep the instance alive between main() calls

def init():
    global inference_interface
    base_output_dir = Path(REX.p0.v).expanduser().resolve()
    model_name = str(REX.p1.v) if isinstance(REX.p1.v, (bool, float, int)) else REX.p1.v
    runtime_name = str(REX.p2.v) if isinstance(REX.p2.v, (bool, float, int)) else REX.p2.v

    REX.TraceInfo(f"Initializing inference with runtime: {runtime_name}, model: {model_name}")
    inference_interface = InferenceInterface(base_output_dir, runtime_name, model_name)

def exit():
    global inference_interface
    if inference_interface is not None:
        inference_interface.release()

def main():
    global inference_interface
    if inference_interface is not None:
        inference_interface.run_inference()

class InferenceInterface:
    def __init__(self, save_dir="results", runtime="onnxruntime", model_name="mobilenetv2"):
        """
        Initialize the inference interface

        Args:
            save_dir (str): Directory to save results
            runtime (str): Runtime to use ('onnxruntime' or 'opencv')
            model_name (str): Model to use ('mobilenetv2', 'squeezenet', 'resnet50', etc.)
        """
        REX.TraceInfo(f"Initializing InferenceInterface with runtime: {runtime}, model: {model_name}")

        self.save_dir = save_dir
        self.runtime = runtime
        self.model_name = model_name
        self.model = None
        self.input_name = None
        self.results_file = os.path.join(save_dir, f"{runtime}_{model_name}_results.csv")
        self.data_dir = os.path.join(save_dir, "data")

        # Create output directory if it doesn't exist
        self.create_folder_if_not_exists(save_dir)
        self.create_folder_if_not_exists(self.data_dir)

        # Initialize CSV file if it doesn't exist
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Runtime', 'Model', 'Image', 'Inference Time (ms)', 'Top Prediction', 'Probability'])

        # Load the model
        self.load_model()

        # Prepare sample input data (random data for testing)
        self.prepare_sample_data()

    def load_model(self):
        """Load the selected model with the selected runtime"""
        try:
            # Define model paths for models in models directory
            model_paths = {
                'mobilenetv2': 'root/models/mobilenetv2_100_Opset16.onnx',
                'squeezenet': 'root/models/squeezenet1_0_Opset17.onnx',
                'resnet50': 'root/models/resnet50_Opset17.onnx',
                'efficientnet_b0': 'root/models/efficientnet_b0_Opset17.onnx',
                'vit_b16': 'root/models/vit_b_16_Opset17.onnx'
            }

            REX.TraceInfo(f"Loading model: {self.model_name} with runtime: {self.runtime}")

            if self.model_name in model_paths:
                model_path = model_paths[self.model_name]
                REX.TraceInfo(f"Model path: {model_path}")

                # Set input shapes based on the model
                if self.model_name == 'mobilenetv2':
                    self.input_shape = (1, 3, 224, 224)
                elif self.model_name == 'squeezenet':
                    self.input_shape = (1, 3, 224, 224)
                elif self.model_name == 'resnet50':
                    self.input_shape = (1, 3, 224, 224)
                elif self.model_name == 'efficientnet_b0':
                    self.input_shape = (1, 3, 224, 224)
                elif self.model_name == 'vit_b16':
                    self.input_shape = (1, 3, 224, 224)

                # Load the actual model based on the runtime
                if self.runtime == 'onnxruntime':
                    REX.TraceInfo(f"Creating ONNX Runtime session for {self.model_name}")
                    try:
                        # Create ONNX Runtime inference session
                        self.model = ort.InferenceSession(model_path)

                        # Get input name from the model
                        model_inputs = self.model.get_inputs()
                        if model_inputs and len(model_inputs) > 0:
                            self.input_name = model_inputs[0].name
                            REX.TraceInfo(f"Detected input name from model: {self.input_name}")
                        else:
                            REX.TraceError("Could not detect input name from model")
                            return

                        REX.TraceInfo(f"Successfully loaded model with ONNX Runtime")
                    except Exception as e:
                        REX.TraceError(f"Error loading model with ONNX Runtime: {e}")
                        return
                elif self.runtime == 'opencv':
                    REX.TraceInfo(f"Loading model with OpenCV DNN for {self.model_name}")
                    try:
                        # Load model with OpenCV DNN
                        self.model = cv2.dnn.readNetFromONNX(model_path)

                        # For OpenCV DNN, we need to know the input name
                        # Since OpenCV doesn't provide a way to get input names,
                        # we'll use a default name or try to get it from the ONNX model directly
                        try:
                            import onnx
                            onnx_model = onnx.load(model_path)
                            if onnx_model.graph.input and len(onnx_model.graph.input) > 0:
                                self.input_name = onnx_model.graph.input[0].name
                                REX.TraceInfo(f"Detected input name from ONNX model: {self.input_name}")
                            else:
                                # Default input name for OpenCV DNN
                                self.input_name = ""  # OpenCV doesn't use input names directly
                                REX.TraceInfo("Using default empty input name for OpenCV DNN")
                        except Exception as e:
                            # Default input name for OpenCV DNN
                            self.input_name = ""  # OpenCV doesn't use input names directly
                            REX.TraceInfo(f"Could not detect input name from ONNX model: {e}. Using default empty input name for OpenCV DNN")

                        REX.TraceInfo(f"Successfully loaded model with OpenCV DNN")
                    except Exception as e:
                        REX.TraceError(f"Error loading model with OpenCV DNN: {e}")
                        return
                else:
                    REX.TraceError(f"Unsupported runtime: {self.runtime}")
            else:
                REX.TraceError(f"Unsupported model: {self.model_name}")
        except Exception as e:
            REX.TraceError(f"Error loading model: {e}")

    def prepare_sample_data(self):
        """Prepare sample input data for inference using real images"""
        try:
            # Download sample images if they don't exist
            REX.TraceInfo("Downloading sample images...")
            sample_images = sample_data.download_sample_images(self.data_dir)

            if not sample_images:
                REX.TraceError("Failed to download sample images, using random data instead")
                # Fall back to random data
                if hasattr(self, 'input_shape'):
                    self.sample_input = np.random.random(self.input_shape).astype(np.float32)
                else:
                    self.input_shape = (1, 3, 224, 224)
                    self.sample_input = np.random.random(self.input_shape).astype(np.float32)
                self.sample_image_name = "random_data"
                return

            # Use cat image as default sample
            if 'cat' in sample_images:
                sample_image_path = sample_images['cat']
                self.sample_image_name = "cat"
            else:
                # Use the first available image
                self.sample_image_name = list(sample_images.keys())[0]
                sample_image_path = sample_images[self.sample_image_name]

            REX.TraceInfo(f"Using {self.sample_image_name} image for inference: {sample_image_path}")

            # Preprocess the image
            if hasattr(self, 'input_shape'):
                input_shape = (self.input_shape[2], self.input_shape[3])  # Height, Width
            else:
                input_shape = (224, 224)
                self.input_shape = (1, 3, 224, 224)

            # Preprocess the image using our sample_data module
            self.sample_input = sample_data.preprocess_image(
                sample_image_path,
                input_shape=input_shape
            )

            REX.TraceInfo(f"Prepared sample image with shape: {self.sample_input.shape}")

        except Exception as e:
            REX.TraceError(f"Error preparing sample data: {e}")
            # Fall back to random data
            if hasattr(self, 'input_shape'):
                self.sample_input = np.random.random(self.input_shape).astype(np.float32)
            else:
                self.input_shape = (1, 3, 224, 224)
                self.sample_input = np.random.random(self.input_shape).astype(np.float32)
            self.sample_image_name = "random_data"
            REX.TraceInfo(f"Using random data with shape: {self.sample_input.shape}")

    def run_inference(self):
        """Run inference with the loaded model and measure time"""
        if self.model is None:
            REX.TraceError("Model not loaded")
            return

        try:
            REX.TraceInfo(f"Running inference with {self.runtime} on {self.model_name}")

            # Measure inference time
            start_time = time.time()

            # Run actual inference based on the runtime
            if self.runtime == 'onnxruntime':
                try:
                    # Run inference with ONNX Runtime
                    output = self.model.run(None, {self.input_name: self.sample_input})
                    REX.TraceInfo(f"ONNX Runtime inference successful")
                except Exception as e:
                    REX.TraceError(f"Error during ONNX Runtime inference: {e}")
                    return
            elif self.runtime == 'opencv':
                try:
                    # Run inference with OpenCV DNN
                    self.model.setInput(self.sample_input)
                    output = self.model.forward()
                    REX.TraceInfo(f"OpenCV DNN inference successful")
                except Exception as e:
                    REX.TraceError(f"Error during OpenCV DNN inference: {e}")
                    return

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

            REX.TraceInfo(f"Inference completed in {inference_time:.2f} ms")

            # Process output and get top predictions
            top_predictions = None

            if self.runtime == 'onnxruntime':
                if output and len(output) > 0:
                    REX.TraceInfo(f"Output shape: {output[0].shape}")

                    # Get top-5 predictions
                    top_predictions = sample_data.get_top_predictions(output[0], top_k=5)

                    # Log the predictions
                    REX.TraceInfo(f"Top predictions for {self.sample_image_name}:")
                    for i, (class_id, class_name, prob) in enumerate(top_predictions):
                        # Format the output to be more readable in the log
                        REX.TraceInfo(f"  {i+1}. Class {class_id}: '{class_name}' with probability {prob:.4f}")

            elif self.runtime == 'opencv':
                if output is not None:
                    REX.TraceInfo(f"Output shape: {output.shape}")

                    # Get top-5 predictions
                    top_predictions = sample_data.get_top_predictions(output, top_k=5)

                    # Log the predictions
                    REX.TraceInfo(f"Top predictions for {self.sample_image_name}:")
                    for i, (class_id, class_name, prob) in enumerate(top_predictions):
                        # Format the output to be more readable in the log
                        REX.TraceInfo(f"  {i+1}. Class {class_id}: '{class_name}' with probability {prob:.4f}")

            # Log the result with top prediction
            self.log_result(inference_time, top_predictions)

        except Exception as e:
            REX.TraceError(f"Error during inference: {e}")

    def log_result(self, inference_time, top_prediction=None):
        """Log the inference result to CSV file"""
        try:
            # Get the sample image name if available
            image_name = getattr(self, 'sample_image_name', 'unknown')

            # Extract top prediction and probability if provided
            top_class = "unknown"
            top_prob = 0.0

            if top_prediction and len(top_prediction) > 0:
                # top_prediction is a list of (class_id, class_name, probability) tuples
                _, top_class, top_prob = top_prediction[0]

            with open(self.results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.runtime, self.model_name, image_name, f"{inference_time:.2f}", top_class, f"{top_prob:.4f}"])
            REX.TraceInfo(f"Result logged to {self.results_file} with top prediction: {top_class} ({top_prob:.4f})")
        except Exception as e:
            REX.TraceError(f"Error logging result: {e}")

    def release(self):
        """Release resources"""
        REX.TraceInfo("Releasing InferenceInterface resources")
        # In a real scenario, you might need to release resources here
        self.model = None

    def create_folder_if_not_exists(self, folder_path):
        """Create a folder if it doesn't exist"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            REX.TraceInfo(f"Created folder: {folder_path}")
        return folder_path
