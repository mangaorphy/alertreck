"""
TFLite Model Inference Module
==============================
Loads and runs TensorFlow Lite model for audio classification.
Optimized for Raspberry Pi edge inference.
"""

import numpy as np
from typing import Optional, Tuple
import time

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from ..config import MODEL_PATH, CLASS_NAMES, DEBUG_MODE, MAX_INFERENCE_TIME
from ..utils import PerformanceTimer


class TFLiteModel:
    """
    TensorFlow Lite model wrapper for audio classification.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize TFLite model.
        
        Args:
            model_path: Path to .tflite model file (uses config default if None)
        """
        self.model_path = model_path or MODEL_PATH
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = CLASS_NAMES
        
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model and get input/output details."""
        try:
            print(f"📥 Loading model: {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            # Load TFLite model
            self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("✅ Model loaded successfully!")
            print(f"  Input shape: {self.input_details[0]['shape']}")
            print(f"  Input dtype: {self.input_details[0]['dtype']}")
            print(f"  Output shape: {self.output_details[0]['shape']}")
            print(f"  Output dtype: {self.output_details[0]['dtype']}")
            print(f"  Classes: {self.class_names}")
        
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Optional[Tuple[int, float, np.ndarray]]:
        """
        Run inference on input data.
        
        Args:
            input_data: Preprocessed audio (mel spectrogram)
            
        Returns:
            Tuple of (predicted_class_idx, confidence, all_probabilities)
            or None on error
        """
        try:
            with PerformanceTimer("Inference", warn_threshold=MAX_INFERENCE_TIME):
                # Ensure input has batch dimension
                if len(input_data.shape) == 3:
                    input_data = np.expand_dims(input_data, axis=0)
                
                # Convert to expected dtype
                input_dtype = self.input_details[0]['dtype']
                if input_data.dtype != input_dtype:
                    input_data = input_data.astype(input_dtype)
                
                # Check input shape
                expected_shape = tuple(self.input_details[0]['shape'])
                if input_data.shape != expected_shape:
                    if DEBUG_MODE:
                        print(f"⚠️  Input shape mismatch: {input_data.shape} vs {expected_shape}")
                    # Try to reshape
                    try:
                        input_data = input_data.reshape(expected_shape)
                    except:
                        print(f"❌ Cannot reshape {input_data.shape} to {expected_shape}")
                        return None
                
                # Set input tensor
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output tensor
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                # Get probabilities (squeeze batch dimension)
                probabilities = output_data[0]
                
                # Get predicted class and confidence
                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_class])
                
                if DEBUG_MODE:
                    print(f"Prediction: class={self.class_names[predicted_class]}, "
                          f"confidence={confidence:.3f}")
                    print(f"All probabilities: {probabilities}")
                
                return predicted_class, confidence, probabilities
        
        except Exception as e:
            print(f"❌ Inference error: {e}")
            return None
    
    def predict_batch(self, input_batch: np.ndarray) -> list:
        """
        Run inference on batch of inputs.
        
        Args:
            input_batch: Batch of preprocessed audio
            
        Returns:
            List of (class_idx, confidence, probabilities) tuples
        """
        results = []
        
        for input_data in input_batch:
            result = self.predict(input_data)
            if result:
                results.append(result)
        
        return results
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"UNKNOWN_{class_idx}"
    
    def get_input_shape(self) -> tuple:
        """Get expected input shape."""
        return tuple(self.input_details[0]['shape'])
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'model_path': str(self.model_path),
            'input_shape': self.get_input_shape(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_shape': tuple(self.output_details[0]['shape']),
            'output_dtype': str(self.output_details[0]['dtype']),
            'num_classes': len(self.class_names),
            'class_names': self.class_names
        }


def test_model():
    """Test the TFLite model."""
    print("\n🤖 Testing TFLiteModel...")
    print("=" * 60)
    
    try:
        # Load model
        model = TFLiteModel()
        
        # Show model info
        info = model.get_model_info()
        print("\n📋 Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Create dummy input matching expected shape
        input_shape = model.get_input_shape()
        print(f"\n🔧 Creating test input: {input_shape}")
        
        # Remove batch dimension for test input creation
        test_shape = input_shape[1:] if input_shape[0] == 1 else input_shape
        test_input = np.random.randn(*test_shape).astype(np.float32)
        
        # Run inference
        print("\n🚀 Running inference...")
        result = model.predict(test_input)
        
        if result:
            class_idx, confidence, probabilities = result
            class_name = model.get_class_name(class_idx)
            
            print("\n✅ Inference successful!")
            print(f"  Predicted class: {class_name} (index: {class_idx})")
            print(f"  Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            print(f"\n  All probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"    {model.class_names[i]}: {prob:.3f} ({prob*100:.1f}%)")
        else:
            print("\n❌ Inference failed!")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_model()
