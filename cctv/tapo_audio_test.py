#!/usr/bin/env python3
"""
Tapo Camera - LOCAL speaker test
No cloud. No admin. Pure local camera account auth.

Credentials : fasspay / fasspay2025
Camera IP   : 192.168.122.129

Requirements:
    pip install pytapo

How to set up camera account (one-time, in Tapo app):
    Camera → Settings → Advanced Settings → Camera Account
    Username : fasspay
    Password : fasspay2025
"""

import time
from pytapo import Tapo

# ── Config ────────────────────────────────────────────────────────────────────
HOST            = "192.168.122.129"
USER            = "fasspay"
PASSWORD        = "fasspay2025"
ALARM_DURATION  = 5   # seconds to play the siren

# ── Connect (local only, no cloud fallback) ───────────────────────────────────
def connect() -> Tapo:
    print(f"[*] Connecting locally to {HOST} as '{USER}' ...")

    # password_cloud=None  → disables cloud auth entirely
    # pytapo will negotiate KLAP (SHA256) or MD5 directly with the device
    tapo = Tapo(
        host           = HOST,
        user           = USER,
        password       = PASSWORD,
        password_cloud = None,   # <-- key: no cloud, purely local
    )
    return tapo


# ── Print device info ─────────────────────────────────────────────────────────
def print_info(tapo: Tapo):
    try:
        info   = tapo.getBasicInfo()
        device = info["device_info"]["basic_info"]
        print(f"[+] Auth OK!")
        print(f"    Model    : {device.get('device_type','N/A')}")
        print(f"    FW       : {device.get('sw_version','N/A')}")
        print(f"    Dev ID   : {device.get('dev_id','N/A')}")
    except Exception as e:
        print(f"[~] Info fetch failed (non-fatal): {e}")


# ── Speaker test ──────────────────────────────────────────────────────────────
def speaker_test(tapo: Tapo):
    print(f"\n[*] Firing speaker alarm for {ALARM_DURATION}s ...")

    # Primary method (works on most models)
    try:
        tapo.startManualAlarm()
        print("[+] Alarm started ✓")
    except Exception as e:
        print(f"[~] Primary method failed: {e}")
        print("[*] Trying direct performRequest fallback ...")
        # Fallback: raw request — works on some older firmware
        try:
            tapo.performRequest({
                "method"    : "do",
                "msg_alarm" : {"manual_msg_alarm": {"action": "start"}},
            })
            print("[+] Alarm started via fallback ✓")
        except Exception as e2:
            raise RuntimeError(f"Both alarm methods failed: {e2}") from e2

    # Count down
    for i in range(ALARM_DURATION, 0, -1):
        print(f"    ♪ {i}s ...", end="\r", flush=True)
        time.sleep(1)

    print()
    tapo.stopManualAlarm()
    print("[+] Alarm stopped ✓")
    print("\n[✓] Speaker test complete — camera is working locally!")


# ── Wait helper (if still suspended) ─────────────────────────────────────────
def handle_suspension(error_msg: str):
    import re
    match = re.search(r"(\d+)\s*second", error_msg)
    if match:
        secs = int(match.group(1))
        mins = secs // 60
        print(f"\n[!] Camera is still suspended.")
        print(f"    Wait {mins}m {secs % 60}s before retrying.")
        print(f"\n    Why this happens:")
        print(f"    The camera locks out after repeated failed logins.")
        print(f"    The suspension is enforced locally on the device.")
        print(f"\n    While you wait, confirm in the Tapo app:")
        print(f"    Settings → Advanced Settings → Camera Account")
        print(f"    Username : {USER}")
        print(f"    Password : {PASSWORD}")
    else:
        print(f"\n[!] Auth error: {error_msg}")
        print(f"\n    Checklist:")
        print(f"    ✗ Wrong credentials? → Verify camera account in Tapo app")
        print(f"    ✗ Camera account not created? → Create it in the app first")
        print(f"    ✗ Older firmware? → Some cameras only accept 'admin' as user")
        print(f"\n    If camera account doesn't work, try:")
        print(f"    USER     = 'admin'")
        print(f"    PASSWORD = '<your TP-Link cloud password>'")
        print(f"    (This still authenticates locally — cloud pw is used as local secret)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        tapo = connect()
        print_info(tapo)
        speaker_test(tapo)

    except Exception as e:
        msg = str(e)
        if "Suspension" in msg or "suspension" in msg or "seconds" in msg.lower():
            handle_suspension(msg)
        else:
            print(f"\n[!] Unexpected error: {e}")
            raise