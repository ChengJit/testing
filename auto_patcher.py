#!/usr/bin/env python3
"""
Auto-patcher for inventory_door_cctv.py
Applies all fixes automatically with backup
"""

import os
import json
import shutil
from datetime import datetime


def backup_files():
    """Backup original files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    files_to_backup = ["inventory_door_cctv.py", "door_config.json"]
    
    for filename in files_to_backup:
        if os.path.exists(filename):
            backup = f"{filename}.backup_{timestamp}"
            shutil.copy2(filename, backup)
            print(f"✅ Backed up: {backup}")


def patch_config():
    """Patch door_config.json"""
    config_file = "door_config.json"
    
    if not os.path.exists(config_file):
        print(f"⚠️  {config_file} not found, skipping")
        return
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    # Add camera resolution if not present
    if "camera_resolution" not in config:
        config["camera_resolution"] = [1280, 720]
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Updated {config_file} with camera_resolution: [1280, 720]")
    else:
        print(f"ℹ️  camera_resolution already in {config_file}")


def patch_python_file():
    """Patch inventory_door_cctv.py"""
    filename = "inventory_door_cctv.py"
    
    if not os.path.exists(filename):
        print(f"❌ {filename} not found!")
        return False
    
    with open(filename, "r") as f:
        content = f.read()
    
    changes_made = 0
    
    # Fix 1: Lower threshold from 0.6 to 0.4
    if "if similarity > 0.6:" in content:
        content = content.replace(
            "if similarity > 0.6:",
            "if similarity > 0.4:  # ← LOWERED from 0.6"
        )
        changes_made += 1
        print("✅ Fix 1: Lowered recognition threshold to 40%")
    
    # Fix 2: Lower close match threshold
    if "if similarity > 0.4:" in content and "# Log close" in content:
        # Find the specific line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "# Log close but not enough matches" in line:
                # Check next line
                if i + 1 < len(lines) and "if similarity > 0.4:" in lines[i+1]:
                    lines[i+1] = lines[i+1].replace(
                        "if similarity > 0.4:",
                        "if similarity > 0.3:  # ← LOWERED"
                    )
                    content = '\n'.join(lines)
                    changes_made += 1
                    print("✅ Fix 2: Lowered close match threshold to 30%")
                    break
    
    # Fix 3: Update training collection threshold
    if "if similarity < 0.4:  # Different person" in content:
        content = content.replace(
            "if similarity < 0.4:  # Different person",
            "if similarity < 0.35:  # Different person ← LOWERED"
        )
        changes_made += 1
        print("✅ Fix 3: Updated training threshold to 35%")
    
    # Fix 4: Add shape fixing before face_distance
    if "distances = face_recognition.face_distance(\n                        self.known_encodings, encoding\n                    )" in content:
        old_code = """distances = face_recognition.face_distance(
                        self.known_encodings, encoding
                    )"""
        
        new_code = """# FIX: Ensure all encodings are proper 1D arrays
                    known_encodings_fixed = []
                    for enc in self.known_encodings:
                        if isinstance(enc, np.ndarray):
                            if len(enc.shape) > 1:
                                enc = enc.flatten()
                            known_encodings_fixed.append(enc)
                        else:
                            enc = np.array(enc, dtype=np.float64).flatten()
                            known_encodings_fixed.append(enc)
                    
                    # Ensure new encoding is 1D
                    if len(encoding.shape) > 1:
                        encoding = encoding.flatten()
                    
                    distances = face_recognition.face_distance(
                        known_encodings_fixed, encoding
                    )"""
        
        content = content.replace(old_code, new_code)
        changes_made += 1
        print("✅ Fix 4: Added embedding shape fixing")
    
    # Fix 5: Update __init__ signature
    if "def __init__(\n        self,\n        rtsp_url,\n        model_size=\"n\",\n        log_file=\"inventory_log.csv\",\n        display_width=1280,\n        embeddings_file=\"known_embeddings.npz\",\n    ):" in content:
        old_sig = """def __init__(
        self,
        rtsp_url,
        model_size="n",
        log_file="inventory_log.csv",
        display_width=1280,
        embeddings_file="known_embeddings.npz",
    ):"""
        
        new_sig = """def __init__(
        self,
        rtsp_url,
        model_size="n",
        log_file="inventory_log.csv",
        display_width=1280,
        embeddings_file="known_embeddings.npz",
        camera_resolution=(1280, 720),  # ← ADDED
    ):"""
        
        content = content.replace(old_sig, new_sig)
        
        # Add camera_resolution assignment
        if "self.display_width = display_width" in content:
            content = content.replace(
                "self.display_width = display_width",
                "self.display_width = display_width\n        self.camera_resolution = camera_resolution  # ← ADDED"
            )
        
        changes_made += 1
        print("✅ Fix 5: Added camera_resolution parameter")
    
    # Fix 6: Set camera resolution in start()
    if "self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer" in content:
        old_code = "self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer"
        new_code = """self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        
        # ← ADDED: Set lower resolution
        if self.camera_resolution:
            width, height = self.camera_resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Requesting resolution: {width}x{height}")"""
        
        content = content.replace(old_code, new_code)
        changes_made += 1
        print("✅ Fix 6: Added camera resolution setting")
    
    # Fix 7: Pass camera_resolution in main
    if """system = OptimizedInventoryDoorCCTV(
        rtsp_url=config["rtsp_url"],
        model_size=config.get("model_size", "n"),
        log_file=config.get("log_file", "inventory_log.csv"),
        display_width=config.get("display_width", 1280),
        embeddings_file=config.get("embeddings_file", "known_embeddings.npz"),
    )""" in content:
        
        old_code = """system = OptimizedInventoryDoorCCTV(
        rtsp_url=config["rtsp_url"],
        model_size=config.get("model_size", "n"),
        log_file=config.get("log_file", "inventory_log.csv"),
        display_width=config.get("display_width", 1280),
        embeddings_file=config.get("embeddings_file", "known_embeddings.npz"),
    )"""
        
        new_code = """system = OptimizedInventoryDoorCCTV(
        rtsp_url=config["rtsp_url"],
        model_size=config.get("model_size", "n"),
        log_file=config.get("log_file", "inventory_log.csv"),
        display_width=config.get("display_width", 1280),
        embeddings_file=config.get("embeddings_file", "known_embeddings.npz"),
        camera_resolution=tuple(config.get("camera_resolution", [1280, 720])),  # ← ADDED
    )"""
        
        content = content.replace(old_code, new_code)
        changes_made += 1
        print("✅ Fix 7: Added camera_resolution parameter in main")
    
    # Save patched file
    if changes_made > 0:
        with open(filename, "w") as f:
            f.write(content)
        print(f"\n✅ Applied {changes_made} fixes to {filename}")
        return True
    else:
        print(f"\n⚠️  No changes needed or patches already applied")
        return False


def main():
    print("\n" + "="*70)
    print("🔧 AUTO-PATCHER FOR FACE RECOGNITION ISSUES")
    print("="*70)
    print("\nThis will automatically fix:")
    print("  1. Lower camera resolution (2880x1620 → 1280x720)")
    print("  2. Lower recognition threshold (60% → 40%)")
    print("  3. Fix embedding shape issues")
    print("  4. Better face matching logic")
    print("="*70)
    
    confirm = input("\nContinue? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled")
        return
    
    print("\n📋 Starting patch process...\n")
    
    # Backup
    print("1️⃣  Creating backups...")
    backup_files()
    
    # Patch config
    print("\n2️⃣  Patching door_config.json...")
    patch_config()
    
    # Patch Python file
    print("\n3️⃣  Patching inventory_door_cctv.py...")
    success = patch_python_file()
    
    print("\n" + "="*70)
    print("✅ PATCHING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Restart your system:")
    print("   python3 inventory_door_cctv.py")
    print("")
    print("2. Press 'D' to enable debug mode")
    print("   This will show face matching scores")
    print("")
    print("3. Expected results:")
    print("   • Camera: 1280x720 (instead of 2880x1620)")
    print("   • Face matches showing 40-60% similarity")
    print("   • People getting recognized")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()