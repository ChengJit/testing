#!/usr/bin/env python3
"""Test speaker via ONVIF (bypasses Tapo cloud completely)"""

import requests
from requests.auth import HTTPDigestAuth

CAMERA_IP = "192.168.122.129"
USER = "fasspay"
PASS = "fasspay2025"

print("=" * 50)
print("  ONVIF SPEAKER TEST")
print("=" * 50)

# ONVIF uses port 2020 on Tapo cameras
ONVIF_URL = f"http://{CAMERA_IP}:2020/onvif/device_service"

# Get device capabilities
soap_envelope = '''<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
    <soap:Body>
        <tds:GetCapabilities>
            <tds:Category>All</tds:Category>
        </tds:GetCapabilities>
    </soap:Body>
</soap:Envelope>'''

print(f"Connecting to {CAMERA_IP} via ONVIF...")

try:
    resp = requests.post(
        ONVIF_URL,
        data=soap_envelope,
        headers={"Content-Type": "application/soap+xml"},
        auth=HTTPDigestAuth(USER, PASS),
        timeout=10
    )
    print(f"Status: {resp.status_code}")

    if "AudioOutput" in resp.text:
        print("Camera supports audio output!")
    if "Backchannel" in resp.text.lower():
        print("Backchannel supported!")

    # Show relevant parts
    if resp.status_code == 200:
        print("\nONVIF response received. Camera is accessible via ONVIF.")
        print("Audio capabilities in response:", "AudioOutput" in resp.text)
    else:
        print(f"Response: {resp.text[:500]}")

except Exception as e:
    print(f"ONVIF error: {e}")
    print("\nTrying port 80...")

    try:
        resp = requests.post(
            f"http://{CAMERA_IP}/onvif/device_service",
            data=soap_envelope,
            headers={"Content-Type": "application/soap+xml"},
            auth=HTTPDigestAuth(USER, PASS),
            timeout=10
        )
        print(f"Port 80 status: {resp.status_code}")
    except Exception as e2:
        print(f"Port 80 also failed: {e2}")
