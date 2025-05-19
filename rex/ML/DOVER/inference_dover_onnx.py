from PyRexExt import REX  # important, don't remove!
import os
import cv2
import numpy as np
import time
import csv
from pathlib import Path
import onnxruntime as ort
import glob
from datetime import datetime
import gc  # Import garbage collector

dover_interface = None  # global variable to keep the instance alive between main() calls

def init():
    global dover_interface
    base_output_dir = Path(REX.p0.v).expanduser().resolve()
    model_name = str(REX.p1.v) if isinstance(REX.p1.v, (bool, float, int)) else REX.p1.v
    runtime_name = str(REX.p2.v) if isinstance(REX.p2.v, (bool, float, int)) else REX.p2.v

    REX.TraceInfo(f"Initializing DOVER inference with runtime: {runtime_name}, model: {model_name}")
    dover_interface = DoverInterface(base_output_dir, runtime_name, model_name)

def exit():
    global dover_interface
    if dover_interface is not None:
        dover_interface.release()

def main():
    global dover_interface
    if dover_interface is not None:
        # Get sequence folder from REX parameter
        sequence_folder = str(REX.p3.v) if isinstance(REX.p3.v, (bool, float, int)) else REX.p3.v
        # Get sequence length from REX parameter
        sequence_length = int(REX.p4.v) if isinstance(REX.p4.v, (bool, float, int)) else int(REX.p4.v)
        sequence_no = str(int(REX.u0.v)) if isinstance(REX.u0.v, (bool, float, int)) else REX.u0.v

        sequence_folder = os.path.join(sequence_folder, sequence_no)
        dover_interface.run_inference(sequence_folder, sequence_length)

class DoverVarLenSequenceDataset:
    """Dataset for loading variable length image sequences for DOVER inference"""

    def __init__(self, folder_path, seq_length=16, transform=None):
        """
        Initialize the dataset

        Args:
            folder_path: Path to folder containing image sequences
            seq_length: Number of frames to use in each sequence
            transform: Optional transform to apply to images
        """
        self.folder_path = folder_path
        self.seq_length = seq_length
        self.transform = transform
        self.image_files = []

        # Find all image files in the folder
        self._find_image_files()

    def _find_image_files(self):
        """Find all image files in the folder and sort them by timestamp"""
        # Look for common image formats
        extensions = ['jpg', 'jpeg', 'png', 'bmp']
        all_files = []

        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(self.folder_path, f"*.{ext}")))

        # Sort files by timestamp in filename (assuming format: timestamp_*.ext)
        self.image_files = sorted(all_files, key=lambda x: os.path.basename(x).split('_')[0])

        # If we don't have enough files, duplicate the last ones
        if len(self.image_files) < self.seq_length:
            REX.TraceInfo(f"Not enough images in folder (found {len(self.image_files)}, need {self.seq_length}). Duplicating last image.")
            while len(self.image_files) < self.seq_length:
                self.image_files.append(self.image_files[-1] if self.image_files else None)

        # Trim to the requested sequence length
        self.image_files = self.image_files[:self.seq_length]

    def load_frames(self):
        """Load the image sequence as frames for DOVER inference"""
        frames = []

        # DOVER-specific normalization values from onnx_inference.py
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        try:
            for img_path in self.image_files:
                if img_path is None:
                    # Create a black frame if no image is available
                    frame = np.zeros((224, 224, 3), dtype=np.float32)
                else:
                    # Read and preprocess the image
                    frame = cv2.imread(img_path)
                    if frame is None:
                        REX.TraceError(f"Failed to read image: {img_path}")
                        frame = np.zeros((224, 224, 3), dtype=np.float32)
                    else:
                        # Resize to 224x224
                        frame = cv2.resize(frame, (224, 224))
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert to float32
                        frame = frame.astype(np.float32)
                        # Explicitly release the original image data
                        del frame

                        # Re-read the image with more memory-efficient approach
                        frame = cv2.imread(img_path)
                        frame = cv2.resize(frame, (224, 224))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = frame.astype(np.float32)

                if self.transform:
                    frame = self.transform(frame)

                # Normalize with DOVER-specific mean and std
                # Note: DOVER doesn't divide by 255 first, it uses the raw pixel values
                frame = (frame - mean) / std

                frames.append(frame)

            return frames
        except Exception as e:
            REX.TraceError(f"Error loading frames: {e}")
            # Clean up in case of exception
            if 'frames' in locals():
                del frames
            if 'frame' in locals():
                del frame
            gc.collect()
            return []

    def prepare_views(self):
        """Prepare aesthetic and technical views for DOVER inference"""
        frames = self.load_frames()

        try:
            # Convert frames to numpy array
            frames_np = np.array(frames, dtype=np.float32)

            # Free memory from original frames list
            del frames

            # Create aesthetic view (single batch)
            # DOVER expects input in format [B, C, T, H, W]
            # B = batch size, C = channels, T = time/frames, H = height, W = width
            aesthetic_view = np.expand_dims(frames_np, axis=0)  # Add batch dimension
            aesthetic_view = np.transpose(aesthetic_view, (0, 4, 1, 2, 3))  # [B, C, T, H, W]

            # Create technical view (4 batches as per DOVER implementation)
            technical_view = np.repeat(np.expand_dims(frames_np, axis=0), 4, axis=0)  # 4 batches
            technical_view = np.transpose(technical_view, (0, 4, 1, 2, 3))  # [B, C, T, H, W]

            # Free memory from frames_np as we no longer need it
            del frames_np

            REX.TraceInfo(f"Created aesthetic view with shape: {aesthetic_view.shape}")
            REX.TraceInfo(f"Created technical view with shape: {technical_view.shape}")

            # Ensure we're using float32 to minimize memory usage
            return aesthetic_view, technical_view
        except Exception as e:
            REX.TraceError(f"Error creating views: {e}")
            # Clean up in case of exception
            if 'frames' in locals():
                del frames
            if 'frames_np' in locals():
                del frames_np
            gc.collect()
            return None, None

class DoverInterface:
    def __init__(self, save_dir="results", runtime="onnxruntime", model_name="dover"):
        """
        Initialize the DOVER inference interface

        Args:
            save_dir (str): Directory to save results
            runtime (str): Runtime to use ('onnxruntime' or 'opencv')
            model_name (str): Model to use ('dover' or 'dover_mobile')
        """
        REX.TraceInfo(f"Initializing DoverInterface with runtime: {runtime}, model: {model_name}")
        # if model_name == "dover_mobile":
        #     raise NotImplementedError("DOVER-Mobile is not supported yet")

        self.save_dir = save_dir
        self.runtime = runtime
        self.model_name = model_name
        self.model = None
        self.results_file = os.path.join(save_dir, f"{runtime}_{model_name}_dover_results.csv")

        # Create output directory if it doesn't exist
        self.create_folder_if_not_exists(save_dir)

        # Initialize CSV file if it doesn't exist
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Runtime', 'Model', 'Sequence', 'Frames', 'Inference Time (ms)', 'Total Time (ms)', 'Aesthetic Score', 'Technical Score', 'Overall Quality Score'])

        # Load the model
        self.load_model()

    def load_model(self):
        """Load the selected DOVER model with the selected runtime"""
        try:
            # Define model paths relative to the current directory
            model_paths = {
                'dover': 'root/models/onnx_dover.onnx',
                'dover_mobile': 'root/models/onnx_dover_mobile.onnx'
            }

            REX.TraceInfo(f"Loading model: {self.model_name} with runtime: {self.runtime}")

            if self.model_name in model_paths:
                model_path = model_paths[self.model_name]
                REX.TraceInfo(f"Model path: {model_path}")

                # Check if the model file exists
                if not os.path.exists(model_path):
                    REX.TraceError(f"Model file not found: {model_path}")
                    return

                # Load the actual model based on the runtime
                if self.runtime == 'onnxruntime':
                    REX.TraceInfo(f"Creating ONNX Runtime session for {self.model_name}")
                    try:
                        # Create ONNX Runtime inference session with optimizations
                        sess_options = ort.SessionOptions()
                        # Enable memory pattern optimization
                        sess_options.enable_mem_pattern = True
                        # Enable memory arena on CPU
                        sess_options.enable_cpu_mem_arena = True
                        # Set graph optimization level
                        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                        self.model = ort.InferenceSession(model_path, sess_options)
                        REX.TraceInfo(f"Successfully loaded model with ONNX Runtime")
                    except Exception as e:
                        REX.TraceError(f"Error loading model with ONNX Runtime: {e}")
                        return
                elif self.runtime == 'opencv':
                    REX.TraceInfo(f"Loading model with OpenCV DNN for {self.model_name}")
                    try:
                        # Load model with OpenCV DNN
                        self.model = cv2.dnn.readNetFromONNX(model_path)
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

    def run_inference(self, sequence_folder, sequence_length=16):
        """Run DOVER inference on a sequence of images"""
        if self.model is None:
            REX.TraceError("Model not loaded")
            return

        start_time = time.time()

        try:
            REX.TraceInfo(f"Running DOVER inference with {self.runtime} on {self.model_name}")
            REX.TraceInfo(f"Sequence folder: {sequence_folder}, Length: {sequence_length}")

            # Create dataset and load sequence
            dataset = DoverVarLenSequenceDataset(sequence_folder, seq_length=sequence_length)
            aesthetic_view, technical_view = dataset.prepare_views()

            if aesthetic_view is None or technical_view is None:
                REX.TraceError("Failed to prepare views for inference")
                return

            # Measure inference time
            inference_start_time = time.time()

            # Run actual inference based on the runtime
            outputs = None
            if self.runtime == 'onnxruntime':
                try:
                    # Run inference with ONNX Runtime
                    outputs = self.model.run(None, {
                        "aes_view": aesthetic_view,
                        "tech_view": technical_view
                    })
                    REX.TraceInfo(f"ONNX Runtime inference successful")
                except Exception as e:
                    REX.TraceError(f"Error during ONNX Runtime inference: {e}")
                    # Clean up memory before returning
                    del aesthetic_view
                    del technical_view
                    gc.collect()
                    return
            elif self.runtime == 'opencv':
                try:
                    # OpenCV doesn't support multiple inputs directly, so we need to handle this differently
                    REX.TraceError("OpenCV runtime is not fully supported for DOVER models with multiple inputs")
                    # Clean up memory before returning
                    del aesthetic_view
                    del technical_view
                    gc.collect()
                    return
                except Exception as e:
                    REX.TraceError(f"Error during OpenCV DNN inference: {e}")
                    # Clean up memory before returning
                    del aesthetic_view
                    del technical_view
                    gc.collect()
                    return

            end_time = time.time()
            inference_time = (end_time - inference_start_time) * 1000  # Convert to milliseconds
            total_time = (end_time - start_time) * 1000  # Convert to milliseconds

            REX.TraceInfo(f"Inference completed in {inference_time:.2f} ms")

            # Process output (DOVER outputs aesthetic and technical scores)
            aesthetic_score = None
            technical_score = None
            overall_score = None

            if self.runtime == 'onnxruntime' and outputs:
                if len(outputs) >= 2:
                    # Extract scores and immediately convert to Python float to avoid keeping numpy arrays
                    aesthetic_score = float(np.mean(outputs[0]))
                    technical_score = float(np.mean(outputs[1]))

                    # Calculate overall quality score using the 4-parameter sigmoid rescaling from DOVER
                    x = (aesthetic_score - 0.1107) / 0.07355 * 0.6104 + (technical_score + 0.08285) / 0.03774 * 0.3896
                    overall_score = float(1 / (1 + np.exp(-x)))

                    REX.TraceInfo(f"Aesthetic quality score: {aesthetic_score:.4f}")
                    REX.TraceInfo(f"Technical quality score: {technical_score:.4f}")
                    REX.TraceInfo(f"Overall quality score: {overall_score:.4f}")

                # Explicitly delete outputs to free memory
                del outputs

            # Log the result
            sequence_name = os.path.basename(sequence_folder)
            self.log_result(inference_time, total_time, sequence_name, sequence_length,
                           aesthetic_score, technical_score, overall_score)

            # Clean up large arrays to free memory
            del aesthetic_view
            del technical_view
            del dataset

            # Force garbage collection
            gc.collect()
            REX.TraceInfo("Memory cleanup completed")

        except Exception as e:
            REX.TraceError(f"Error during DOVER inference: {e}")
            # Clean up memory in case of exception
            gc.collect()

    def log_result(self, inference_time, total_time, sequence_name, sequence_length,
                  aesthetic_score, technical_score, overall_score):
        """Log the DOVER inference result to CSV file"""
        try:
            with open(self.results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.runtime,
                    self.model_name,
                    sequence_name,
                    sequence_length,
                    f"{inference_time:.2f}",
                    f"{total_time:.2f}",
                    f"{aesthetic_score:.4f}" if aesthetic_score is not None else "N/A",
                    f"{technical_score:.4f}" if technical_score is not None else "N/A",
                    f"{overall_score:.4f}" if overall_score is not None else "N/A"
                ])
            REX.TraceInfo(f"Result logged to {self.results_file}")
        except Exception as e:
            REX.TraceError(f"Error logging result: {e}")

    def release(self):
        """Release resources"""
        REX.TraceInfo("Releasing DoverInterface resources")
        if self.model is not None:
            # Set model to None to release the reference
            self.model = None
            # Force garbage collection to clean up memory
            gc.collect()
            REX.TraceInfo("Model resources released and garbage collection performed")

    def create_folder_if_not_exists(self, folder_path):
        """Create a folder if it doesn't exist"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            REX.TraceInfo(f"Created folder: {folder_path}")
        return folder_path
