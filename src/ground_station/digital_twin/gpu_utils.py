"""
GPU Acceleration & High-Performance System Optimization Utilities.
Configures Windows High-Performance GPU preferences, NVIDIA driver acceleration,
and sub-millisecond timer resolution for smooth 60+ FPS rendering.
"""

import os
import sys
import ctypes
from typing import Optional, Dict, Any

IS_WINDOWS = sys.platform == "win32"

def setup_high_performance_gpu() -> Dict[str, Any]:
    """
    Configures Windows DirectX / WGL GPU preference registry and environment
    variables to ensure Python uses the discrete High-Performance GPU (e.g. NVIDIA RTX).
    Returns info dict about detected GPU and setup status.
    """
    status = {
        "gpu_preference_set": False,
        "renderer": "Unknown",
        "vendor": "Unknown",
        "is_discrete": False,
    }

    # Set driver hint environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

    if IS_WINDOWS:
        try:
            import winreg
            # Registry path for Windows Graphics Performance Preferences
            reg_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
            
            # GpuPreference=2 means "High Performance GPU" (NVIDIA / discrete)
            # GpuPreference=1 means "Power Saving GPU" (Integrated Intel / AMD)
            pref_value = "GpuPreference=2;"
            
            # Executables to configure
            exes = set()
            if sys.executable:
                exes.add(os.path.abspath(sys.executable))
            if hasattr(sys, "base_exec_prefix"):
                base_python = os.path.join(sys.base_exec_prefix, "python.exe")
                if os.path.exists(base_python):
                    exes.add(os.path.abspath(base_python))
            if hasattr(sys, "prefix"):
                prefix_python = os.path.join(sys.prefix, "python.exe")
                if os.path.exists(prefix_python):
                    exes.add(os.path.abspath(prefix_python))
                scripts_python = os.path.join(sys.prefix, "Scripts", "python.exe")
                if os.path.exists(scripts_python):
                    exes.add(os.path.abspath(scripts_python))

            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
                for exe in exes:
                    try:
                        winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, pref_value)
                    except Exception:
                        pass
            status["gpu_preference_set"] = True
        except Exception as e:
            status["gpu_preference_error"] = str(e)

    # Detect active OpenGL GPU if possible
    try:
        import glfw
        import OpenGL.GL as gl
        if glfw.init():
            glfw.window_hint(glfw.VISIBLE, False)
            win = glfw.create_window(64, 64, "gpu_detect", None, None)
            if win:
                glfw.make_context_current(win)
                vendor_bytes = gl.glGetString(gl.GL_VENDOR)
                renderer_bytes = gl.glGetString(gl.GL_RENDERER)
                if vendor_bytes:
                    status["vendor"] = vendor_bytes.decode("utf-8", errors="ignore")
                if renderer_bytes:
                    status["renderer"] = renderer_bytes.decode("utf-8", errors="ignore")
                if "nvidia" in status["vendor"].lower() or "nvidia" in status["renderer"].lower() or "geforce" in status["renderer"].lower():
                    status["is_discrete"] = True
                glfw.destroy_window(win)
            glfw.terminate()
    except Exception:
        pass

    return status


def enable_high_resolution_timer() -> bool:
    """Enables 1ms timer resolution on Windows to avoid time.sleep frame drop jitter."""
    if IS_WINDOWS:
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            return True
        except Exception:
            return False
    return False


def disable_high_resolution_timer() -> None:
    """Restores default Windows timer resolution."""
    if IS_WINDOWS:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass

