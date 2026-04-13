#!/usr/bin/env python3
"""Test pytapo with cloud credentials"""

from pytapo import Tapo

CAMERA_IP = "192.168.122.129"

# Cloud credentials
EMAIL = "cjgiam97@gmail.com"
PASSWORD = "Cj6179970216!@"

print("=" * 50)
print("  PYTAPO CLOUD AUTH TEST")
print("=" * 50)
print(f"Camera IP: {CAMERA_IP}")
print(f"Email: {EMAIL}")

try:
    print("\nConnecting...")
    t = Tapo(CAMERA_IP, EMAIL, PASSWORD)
    print("Connected!")

    print("\nGetting camera info...")
    info = t.getBasicInfo()
    print(f"Info: {info}")

    print("\nTriggering alarm sound...")
    t.setAlarm(True)
    print("ALARM ON! Listen to camera speaker!")

    input("\nPress Enter to turn off alarm...")
    t.setAlarm(False)
    print("Alarm off")

except Exception as e:
    print(f"Error: {e}")
