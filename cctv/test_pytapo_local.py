#!/usr/bin/env python3
"""Test pytapo with LOCAL camera account (bypasses cloud)"""

from pytapo import Tapo

CAMERA_IP = "192.168.122.129"

print("=" * 50)
print("  PYTAPO LOCAL AUTH TEST")
print("=" * 50)

# Camera account credentials (set in Tapo app > Advanced Settings > Camera Account)
# NOT the cloud/email credentials
camera_user = "fasspay"
camera_pass = "fasspay2025"

print(f"Camera IP: {CAMERA_IP}")
print(f"Using camera account: {camera_user}")

try:
    # Try with superSecretKey parameter to force local auth
    t = Tapo(
        CAMERA_IP,
        camera_user,
        camera_pass,
        superSecretKey="Z+EvSuEJRopbDHAHiAMa6w=="  # Default Tapo key
    )
    print("Connected!")

    info = t.getBasicInfo()
    print(f"Camera model: {info}")

    print("\nTriggering alarm...")
    t.setAlarm(True)
    print("ALARM ON - listen to camera!")

    input("Press Enter to stop alarm...")
    t.setAlarm(False)
    print("Alarm off")

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative method...")

    try:
        # Some versions use different params
        from pytapo import Tapo
        t = Tapo(CAMERA_IP, camera_user, camera_pass)
        t.setAlarm(True)
        print("Alarm triggered!")
    except Exception as e2:
        print(f"Also failed: {e2}")
