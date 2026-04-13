#!/usr/bin/env python3
"""Test pytapo connection and speaker"""

from pytapo import Tapo

CAMERA_IP = "192.168.122.129"

print("=" * 50)
print("  PYTAPO SPEAKER TEST")
print("=" * 50)

email = input("Tapo email: ").strip()
password = input("Tapo password: ").strip()

print(f"\nConnecting to {CAMERA_IP}...")

try:
    t = Tapo(CAMERA_IP, email, password)
    print("Connected!")

    info = t.getBasicInfo()
    print(f"Camera: {info.get('device_info', {}).get('basic_info', {}).get('device_alias', 'Unknown')}")

    print("\nTriggering alarm sound...")
    t.setAlarm(True)
    print("Alarm ON! You should hear it from the camera.")

    input("\nPress Enter to turn off alarm...")
    t.setAlarm(False)
    print("Alarm OFF")

except Exception as e:
    print(f"Error: {e}")
