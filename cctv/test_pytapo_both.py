#!/usr/bin/env python3
"""Test pytapo with camera account + cloud password"""

from pytapo import Tapo

CAMERA_IP = "192.168.122.129"

# Camera account (from Tapo app > Advanced Settings > Camera Account)
CAMERA_USER = "fasspay"
CAMERA_PASS = "fasspay2025"

# Cloud password (your Tapo app login)
CLOUD_PASS = "Cj6179970216!@"

print("=" * 50)
print("  PYTAPO AUTH TEST")
print("=" * 50)
print(f"Camera: {CAMERA_IP}")
print(f"User: {CAMERA_USER}")

# Try different combinations
attempts = [
    ("Camera creds only", CAMERA_USER, CAMERA_PASS, None),
    ("Camera + cloud pass", CAMERA_USER, CAMERA_PASS, CLOUD_PASS),
    ("Cloud email + pass", "cjgiam97@gmail.com", CLOUD_PASS, None),
]

for name, user, pwd, cloud_pwd in attempts:
    print(f"\n[{name}]")
    try:
        if cloud_pwd:
            t = Tapo(CAMERA_IP, user, pwd, cloudPassword=cloud_pwd)
        else:
            t = Tapo(CAMERA_IP, user, pwd)

        print("  Connected!")
        t.setAlarm(True)
        print("  ALARM ON! Check speaker!")

        input("  Press Enter to stop...")
        t.setAlarm(False)
        print("  Done!")
        break

    except Exception as e:
        err = str(e)[:60]
        print(f"  Failed: {err}")
