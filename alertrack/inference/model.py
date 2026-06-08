"""
ONNX Runtime Inference Module
==============================
Loads and runs the exported AlertReck CNN for audio classification.
Optimized for Raspberry Pi edge inference via onnxruntime.
"""

import numpy as np
from typing import Optional, Tuple
import time

import onnxruntime as ort

from ..config import MODEL_PATH, CLASS_NAMES, N_CLASSES, DEBUG_MODE, MAX_INFERENCE_TIME
from ..utils import PerformanceTimer


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ONNXModel:
    """ONNX Runtime wrapper for the AlertReck CNN."""

    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self.class_names = CLASS_NAMES
        self.session: Optional[ort.InferenceSession] = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}\n"
                                    "Run: python3 scripts/export_model.py")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape)
        self.output_name = self.session.get_outputs()[0].name

        print("Model loaded.")
        print(f"  Input  : {self.input_name}  {self.input_shape}")
        print(f"  Output : {self.output_name}")
        print(f"  Classes: {self.class_names}")

    def predict(self, input_data: np.ndarray) -> Optional[Tuple[int, float, np.ndarray]]:
        """
        Run inference on a single mel spectrogram.

        Args:
            input_data: shape (1, 128, 301) or (128, 301) — float32

        Returns:
            (predicted_class_idx, confidence, probabilities[7])
        """
        try:
            with PerformanceTimer("Inference", warn_threshold=MAX_INFERENCE_TIME):
                # Ensure shape is (1, 1, 128, 259) — (batch, channel, mel, time)
                x = input_data.astype(np.float32)
                while x.ndim < 4:
                    x = np.expand_dims(x, axis=0)

                logits = self.session.run(
                    [self.output_name], {self.input_name: x}
                )[0][0]                          # shape (7,)

                probs = _softmax(logits)
                pred  = int(np.argmax(probs))
                conf  = float(probs[pred])

                if DEBUG_MODE:
                    print(f"Prediction: {self.class_names[pred]}  conf={conf:.3f}")

                return pred, conf, probs

        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def get_class_name(self, class_idx: int) -> str:
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"UNKNOWN_{class_idx}"

    def get_input_shape(self) -> tuple:
        return self.input_shape

    def get_model_info(self) -> dict:
        return {
            "model_path":  str(self.model_path),
            "input_shape": self.input_shape,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
        }


# Keep TFLiteModel as an alias so any older code that imports it still works
TFLiteModel = ONNXModel


def test_model():
    print("\nTesting ONNXModel...")
    print("=" * 60)
    try:
        model = ONNXModel()
        test_input = np.random.randn(1, 128, 301).astype(np.float32)
        result = model.predict(test_input)
        if result:
            idx, conf, probs = result
            print(f"Predicted : {model.get_class_name(idx)}  ({conf*100:.1f}%)")
            print("All probabilities:")
            for i, p in enumerate(probs):
                print(f"  [{i}] {model.class_names[i]:<26}  {p:.3f}")
        else:
            print("Inference failed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    print("=" * 60)


if __name__ == "__main__":
    test_model()
