# Touch ID requires pyobjc on macOS:
# pip install pyobjc-framework-LocalAuthentication
#
# If pyobjc is not installed, the fallback PIN dialog is used instead.
# Demo PIN for fallback: 1234

from __future__ import annotations

import sys
import threading
from typing import Optional


HAS_NATIVE_TOUCH_ID = False
if sys.platform == "darwin":
    try:
        import objc  # type: ignore
        from LocalAuthentication import (  # type: ignore
            LAContext,
            LAPolicyDeviceOwnerAuthentication,
            LAPolicyDeviceOwnerAuthenticationWithBiometrics,
        )

        HAS_NATIVE_TOUCH_ID = True
    except Exception:  # pragma: no cover - best effort detection
        HAS_NATIVE_TOUCH_ID = False
        LAContext = None  # type: ignore
        LAPolicyDeviceOwnerAuthentication = None  # type: ignore
        LAPolicyDeviceOwnerAuthenticationWithBiometrics = None  # type: ignore
else:
    LAContext = None  # type: ignore
    LAPolicyDeviceOwnerAuthentication = None  # type: ignore
    LAPolicyDeviceOwnerAuthenticationWithBiometrics = None  # type: ignore


def request_touch_id(reason: str = "Verify your identity") -> bool:
    """
    Trigger macOS Touch ID authentication dialog.
    Returns True if authenticated, False if failed/cancelled/unavailable.
    Blocks until the user responds or times out (30 seconds).
    """
    if HAS_NATIVE_TOUCH_ID:
        try:
            return _macos_touch_id(reason)
        except Exception:
            pass
    return _fallback_pin_dialog(reason)


def _macos_touch_id(reason: str) -> bool:
    """
    Use pyobjc LocalAuthentication framework to trigger Touch ID.
    Requires: pip install pyobjc-framework-LocalAuthentication
    """
    if not HAS_NATIVE_TOUCH_ID or LAContext is None:
        raise RuntimeError("Native Touch ID is not available on this system.")

    result_holder: list[Optional[bool]] = [None]
    event = threading.Event()

    ctx = LAContext()  # type: ignore[call-arg]
    can_evaluate = ctx.canEvaluatePolicy_error_(  # type: ignore[attr-defined]
        LAPolicyDeviceOwnerAuthenticationWithBiometrics,
        None,
    )

    policy = (
        LAPolicyDeviceOwnerAuthenticationWithBiometrics
        if can_evaluate
        else LAPolicyDeviceOwnerAuthentication
    )

    def reply_handler(success, _error):
        result_holder[0] = bool(success)
        event.set()

    ctx.evaluatePolicy_localizedReason_reply_(policy, reason, reply_handler)  # type: ignore[attr-defined]
    event.wait(timeout=30.0)

    if result_holder[0] is None:
        return False
    return bool(result_holder[0])


def _fallback_pin_dialog(reason: str) -> bool:
    """
    Fallback PIN dialog for non-Mac or when pyobjc is unavailable.
    Prefers a tkinter dialog on the main thread, else falls back to console input.
    Hardcoded PIN is "1234" for demo purposes.
    """
    if threading.current_thread() is threading.main_thread():
        try:
            import tkinter as tk
            from tkinter import simpledialog
        except Exception:
            pass
        else:
            demo_pin = "1234"
            root = tk.Tk()
            root.withdraw()
            pin = simpledialog.askstring(
                "Identity Verification Required",
                f"{reason}\n\nEnter PIN to continue:",
                show="*",
                parent=root,
            )
            root.destroy()
            if pin is None:
                return False
            return pin.strip() == demo_pin

    return _fallback_console_pin(reason)


def _fallback_console_pin(reason: str) -> bool:
    """
    Console fallback when no GUI prompt is available.
    """
    try:
        from getpass import getpass

        prompt = f"{reason}\nEnter PIN to continue (demo PIN 1234): "
        pin = getpass(prompt)
    except Exception:
        try:
            pin = input(f"{reason}\nEnter PIN to continue (demo PIN 1234): ")
        except Exception:
            return False
    if pin is None:
        return False
    return pin.strip() == "1234"
