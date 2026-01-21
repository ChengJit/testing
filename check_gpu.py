import onnxruntime as ort
from insightface.app import FaceAnalysis

print("--- ONNX Runtime Check ---")
print(f"Available Providers: {ort.get_available_providers()}")

# Test InsightFace initialization
try:
    # We include Tensorrt for Jetson specifically
    app = FaceAnalysis(providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    print("\n--- InsightFace Results ---")
    print(f"Success! Models are using: {app.models['detection'].session.get_providers()}")
except Exception as e:
    print(f"\nError: {e}")