#!/usr/bin/env python3
"""
INSIGHTFACE-ACCELERATED Face Recognition Training for Jetson
- Generates 'known_embeddings_insightface.npz'
- FULLY COMPATIBLE with inventory_door_cctv.py
- Handles NumPy compatibility issues
"""

import os
import cv2
import numpy as np
from datetime import datetime
import shutil
import gc
import time
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Fix for NumPy compatibility
def fix_numpy_import():
    """Fix NumPy import issues common on Jetson"""
    try:
        # Force reload numpy core modules
        import importlib
        modules_to_reload = [
            'numpy.core.multiarray',
            'numpy.core.umath', 
            'numpy.core._multiarray_umath',
            'numpy'
        ]
        
        for module in modules_to_reload:
            if module in sys.modules:
                del sys.modules[module]
        
        # Reimport
        import numpy as np
        print(f"✅ NumPy fixed: {np.__version__}")
        return np
    except Exception as e:
        print(f"⚠️  NumPy fix attempt failed: {e}")
        import numpy as np
        return np

# Apply numpy fix
np = fix_numpy_import()

def install_package(package_name, version=None):
    """Install package with error handling"""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if version:
            cmd.append(f"{package_name}=={version}")
        else:
            cmd.append(package_name)
        
        cmd.append("--no-cache-dir")
        
        print(f"📦 Installing {package_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {package_name} installed successfully")
            return True
        else:
            print(f"⚠️  Installation output: {result.stdout}")
            return False
    except Exception as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def check_and_install_requirements():
    """Check and install required packages"""
    jetson_packages = {
        'numpy': '1.19.5',
        'insightface': '0.7.3',
        'onnxruntime': '1.14.1',
    }
    
    print("\n" + "="*70)
    print("🔧 CHECKING AND INSTALLING DEPENDENCIES")
    print("="*70)
    
    try:
        import insightface
        print(f"✅ insightface: {insightface.__version__}")
    except ImportError:
        print("❌ insightface not found")
        install_package('insightface', '0.7.3')
    
    try:
        import onnxruntime as ort
        print(f"✅ onnxruntime: {ort.__version__}")
    except ImportError:
        print("❌ onnxruntime not found")
        install_package('onnxruntime', '1.14.1') # GPU version usually preferred
    
    print("="*70)
    return True

def get_insightface_app(use_gpu=True):
    """Get InsightFace app (Buffalo_L for best accuracy)"""
    try:
        from insightface.app import FaceAnalysis
        
        os.makedirs('./models', exist_ok=True)
        
        # Configure providers
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
            print("🔧 Using CUDA (GPU)")
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            print("🔧 Using CPU")
        
        # Use buffalo_l for better accuracy (matches CCTV script)
        app = FaceAnalysis(
            name='buffalo_l', 
            root='./models',
            providers=providers
        )
        
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print(f"✅ InsightFace (Buffalo_L) loaded successfully")
        return app
        
    except Exception as e:
        print(f"❌ Failed to initialize InsightFace: {e}")
        # Fallback
        try:
            print("🔄 Trying CPU-only fallback (Buffalo_S)...")
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name='buffalo_s', root='./models', providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            return app
        except Exception as e2:
            print(f"❌ CPU fallback also failed: {e2}")
            return None

def extract_embeddings_simple(image_path, face_app):
    """Extract normalized embedding from image"""
    embeddings = []
    try:
        img = cv2.imread(image_path)
        if img is None: return []
        
        faces = face_app.get(img)
        if not faces: return []
        
        # Sort by size to get the main face
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        
        # Return the largest face's normed embedding
        # CCTV script uses 'normed_embedding' for cosine similarity
        for face in faces[:1]: 
            embeddings.append(face.normed_embedding)
        
        return embeddings
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return []

def process_person_folder(person_name, person_path, face_app, batch_data):
    """Process images for one person"""
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG']:
        image_files.extend([f for f in os.listdir(person_path) if f.endswith(ext)])
    
    if not image_files:
        print(f"   ⚠️  No images found for {person_name}")
        return 0
    
    print(f"   📸 Found {len(image_files)} images")
    embeddings_count = 0
    
    for img_idx, img_file in enumerate(image_files):
        img_path = os.path.join(person_path, img_file)
        try:
            embeddings = extract_embeddings_simple(img_path, face_app)
            if embeddings:
                for emb in embeddings:
                    # KEY FIX: Ensure it is compatible with CCTV script expectation
                    batch_data['encodings'].append(emb) 
                    batch_data['names'].append(person_name)
                    embeddings_count += 1
                print(f"      ✅ [{img_idx+1}/{len(image_files)}] {img_file}")
            else:
                print(f"      ⚠️  [{img_idx+1}/{len(image_files)}] {img_file}: No face found")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    return embeddings_count

def save_embeddings(embeddings_file, encodings, names):
    """Save embeddings with 'encodings' key for CCTV compatibility"""
    try:
        # Convert to numpy arrays
        encodings_array = np.array(encodings, dtype=np.float32)
        names_array = np.array(names, dtype=object)
        
        # KEY FIX: Use 'encodings' key to match inventory_door_cctv.py
        np.savez_compressed(
            embeddings_file,
            encodings=encodings_array,
            names=names_array
        )
        
        print(f"      💾 Saved {len(encodings)} embeddings to {embeddings_file}")
        return True
    except Exception as e:
        print(f"      ❌ Failed to save: {e}")
        return False

def train_embeddings():
    """Main training routine"""
    check_and_install_requirements()
    
    faces_dir = "known_faces"
    if not os.path.exists(faces_dir):
        print(f"❌ Directory '{faces_dir}' not found. Please create it and add person folders.")
        return False
    
    person_folders = [f for f in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, f))]
    if not person_folders:
        print(f"❌ No person folders found in '{faces_dir}'")
        return False
    
    print(f"\n📊 Found {len(person_folders)} persons to train.")
    
    # GPU Selection
    use_gpu = input("Use GPU acceleration? (y/N): ").strip().lower() == 'y'
    
    # Init App
    face_app = get_insightface_app(use_gpu)
    if not face_app: return False
    
    # Setup Data
    output_file = "known_embeddings.npz"  # MATCHES CCTV SCRIPT
    batch_data = {'encodings': [], 'names': []}
    
    # Backup existing
    if os.path.exists(output_file):
        shutil.copy2(output_file, output_file + ".bak")
        print(f"ℹ️  Backed up existing file to {output_file}.bak")

    print(f"\n🎯 STARTING TRAINING -> {output_file}")
    
    total_embeddings = 0
    
    for person_name in sorted(person_folders):
        print(f"\n👤 Processing: {person_name}")
        count = process_person_folder(person_name, os.path.join(faces_dir, person_name), face_app, batch_data)
        total_embeddings += count
    
    # Save Final
    if batch_data['encodings']:
        save_embeddings(output_file, batch_data['encodings'], batch_data['names'])
        print("\n✅ TRAINING COMPLETE")
        print(f"   Generated {total_embeddings} embeddings for {len(person_folders)} people.")
        return True
    else:
        print("\n❌ No embeddings were generated.")
        return False

def check_existing_embeddings():
    """Verify the generated file"""
    filename = "known_embeddings_insightface.npz"
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found.")
        return

    try:
        data = np.load(filename, allow_pickle=True)
        # Check for 'encodings' first (CCTV format), then 'embeddings' (Old format)
        if 'encodings' in data:
            encodings = data['encodings']
            key_used = 'encodings'
        elif 'embeddings' in data:
            encodings = data['embeddings']
            key_used = 'embeddings'
        else:
            print(f"❌ Invalid file format. Keys found: {list(data.keys())}")
            return

        names = data['names']
        
        print(f"\n📊 FILE ANALYSIS: {filename}")
        print(f"   Key Used: '{key_used}' (CCTV Compatible: {'YES' if key_used=='encodings' else 'NO'})")
        print(f"   Total Embeddings: {len(encodings)}")
        print(f"   Unique Persons: {len(set(names))}")
        
        if len(encodings) > 0:
            print(f"   Dimensions: {encodings[0].shape}")

    except Exception as e:
        print(f"❌ Error reading file: {e}")

def main():
    while True:
        print("\n" + "="*50)
        print("   INSIGHTFACE TRAINER (CCTV COMPATIBLE)")
        print("="*50)
        print("1. Train New Embeddings")
        print("2. Check Existing File")
        print("3. Exit")
        
        choice = input("\nSelect: ").strip()
        
        if choice == "1":
            train_embeddings()
        elif choice == "2":
            check_existing_embeddings()
        elif choice == "3":
            break
        else:
            print("Invalid selection")

if __name__ == "__main__":
    main()