try:
    # tkinterdnd2 provides drag & drop support; make it optional
    import tkinterdnd2 as _tkdnd
    TkinterDnD = _tkdnd.TkinterDnD
    DND_FILES = _tkdnd.DND_FILES
    USE_TKINTER_DND = True
except Exception:
    TkinterDnD = None
    DND_FILES = None
    USE_TKINTER_DND = False
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import threading
import os
import sys
import time
import re
import math
import signal
from collections import deque
try:
    import GPUtil
except Exception:
    GPUtil = None
import platform
import shlex
from subprocess import CalledProcessError
import logging
from logging.handlers import RotatingFileHandler

# Initialize logger early so it can be used during startup checks
_log_dir = os.path.dirname(os.path.abspath(__file__))
_early_log_path = os.path.join(_log_dir, "app.log")
logger = logging.getLogger("CubeDrop")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = RotatingFileHandler(_early_log_path, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
    _fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s: %(message)s')
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(_fmt)
    logger.addHandler(console)

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        # Prefer the directory where this script resides. Using os.getcwd() or
        # os.path.abspath('.') can point to the caller's working directory and
        # cause resource lookups to fail when started from elsewhere.
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# Locate ffmpeg and ffprobe. Prefer bundled copies next to the script (or in
# PyInstaller _MEIPASS). If not present, fall back to executables available on
# PATH.
ffmpeg_path = resource_path("ffmpeg.exe")
ffprobe_path = resource_path("ffprobe.exe")

if not os.path.exists(ffmpeg_path):
    # Try to find ffmpeg on PATH
    from shutil import which
    found = which("ffmpeg")
    if found:
        ffmpeg_path = found
        logger.info("Using ffmpeg from PATH: %s", ffmpeg_path)
    else:
        logger.error("ffmpeg.exe not found in app folder or on PATH: %s", ffmpeg_path)

if not os.path.exists(ffprobe_path):
    from shutil import which
    foundp = which("ffprobe")
    if foundp:
        ffprobe_path = foundp
        logger.info("Using ffprobe from PATH: %s", ffprobe_path)
    else:
        logger.error("ffprobe.exe not found in app folder or on PATH: %s", ffprobe_path)

if not os.path.exists(ffmpeg_path) or not os.path.exists(ffprobe_path):
    messagebox.showerror("Error", "Required FFmpeg files not found. Please ensure ffmpeg.exe and ffprobe.exe are available in the application folder or on your PATH.")
    logger.error("Required FFmpeg files not found. Exiting.")
    sys.exit(1)

# Now that resource_path is defined, ensure the rotating file handler points to
# the application folder (if it differs from the early log path). Replace the
# handler's file if necessary.
try:
    log_path = resource_path("app.log")
    # If the resolved path differs from our early path, switch file handlers.
    if os.path.abspath(log_path) != os.path.abspath(_early_log_path):
        # Remove existing file handlers and add a new one targeting the app folder
        for h in list(logger.handlers):
            logger.removeHandler(h)
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
        fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s: %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(fmt)
        logger.addHandler(console)
except Exception:
    # If logging reconfiguration fails, continue with the existing handlers
    logger.exception("Could not reconfigure logging to app folder")

logger.info("Application start. ffmpeg: %s, ffprobe: %s", ffmpeg_path, ffprobe_path)

process = None
cancel_requested = False
output_file = ""
current_file = ""
encoding_thread = None
encoding_queue = deque()
current_encoding_index = -1
delete_original_var = None
queue_tree = None  # ttk.Treeview showing filename + status
queue_items_map = {}  # filepath -> tree item id
item_to_path = {}     # tree item id -> filepath
batch_results = []
is_batch_processing = False

# Global variables for window dragging
x_offset, y_offset = 0, 0
dragging = False

def detect_available_encoders():
    """Return a list of available encoders on this system.

    Always include CPU (x265). Detect hardware/backends by asking ffmpeg which
    encoders it has and by probing GPUs where possible.
    """
    encoders = ["CPU (x265)"]

    # First, check ffmpeg for encoder support (most reliable for runtime)
    try:
        cmd = [ffmpeg_path, "-hide_banner", "-encoders"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        ffmpeg_out = (result.stdout or "") + (result.stderr or "")
        ffmpeg_out = ffmpeg_out.lower()

        # NVENC
        if 'hevc_nvenc' in ffmpeg_out or 'nvenc' in ffmpeg_out:
            encoders.append("NVENC")

        # Intel QSV
        if 'hevc_qsv' in ffmpeg_out or 'qsv' in ffmpeg_out:
            encoders.append("Intel QSV")

        # AMD AMF
        if 'hevc_amf' in ffmpeg_out or 'amf' in ffmpeg_out:
            encoders.append("AMD AMF")
    except Exception as e:
        logger.warning("ffmpeg probe failed: %s", e)

    # As an extra probe, check GPU presence via GPUtil or platform tools to
    # avoid suggesting encoders when no GPU exists. This won't remove encoders
    # detected via ffmpeg, but will avoid suggesting hardware if clearly absent.
    try:
        if GPUtil:
            gpus = GPUtil.getGPUs() or []
            gpu_names = [g.name.lower() for g in gpus]
        else:
            gpus = []
            gpu_names = []
        if not gpu_names:
            # No GPU detected via GPUtil: on Windows try WMIC as a fallback
            if platform.system() == 'Windows':
                try:
                    wmic = subprocess.run([
                        'wmic', 'path', 'win32_VideoController', 'get', 'name'
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                    out = (wmic.stdout or '').lower()
                    gpu_names = [line.strip() for line in out.splitlines() if line.strip()]
                    gpu_names = [n.lower() for n in gpu_names]
                except Exception:
                    gpu_names = []

        # If GPU list is empty, remove hardware encoders that are only usable if a GPU exists
        if not gpu_names:
            # Keep CPU only
            encoders = [e for e in encoders if e == "CPU (x265)"]
    except Exception:
        # On any error, don't be strict—keep encoders discovered via ffmpeg
        logger.exception("Error while probing GPUs for encoder availability")

    # Ensure deterministic ordering and uniqueness with CPU first
    seen = set()
    ordered = []
    for e in encoders:
        if e not in seen:
            ordered.append(e)
            seen.add(e)

    return ordered

def start_drag(event):
    """Begin window dragging"""
    global x_offset, y_offset, dragging
    dragging = True
    x_offset = event.x
    y_offset = event.y

def stop_drag(event):
    """Stop window dragging"""
    global dragging
    dragging = False

def do_drag(event):
    """Handle window dragging"""
    if dragging:
        x = root.winfo_pointerx() - x_offset
        y = root.winfo_pointery() - y_offset
        root.geometry(f"+{x}+{y}")

def minimize_window():
    """Minimize the window"""
    try:
        # If override-redirect is set (no title bar), some window managers
        # won't allow iconify. Clear it temporarily, iconify, then reapply
        # when the window is restored.
        try:
            is_over = root.overrideredirect()
        except Exception:
            is_over = False

        if is_over:
            try:
                root.overrideredirect(False)
            except Exception:
                logger.exception("Failed to clear overrideredirect before iconify")

            # Iconify (minimize) the window
            try:
                root.iconify()
            except Exception:
                # Fallback: withdraw if iconify fails
                logger.exception("Iconify failed; withdrawing window instead")
                root.withdraw()

            # When the window is mapped (restored), reapply overrideredirect
            def _on_map(event=None):
                try:
                    root.overrideredirect(True)
                except Exception:
                    logger.exception("Could not reapply overrideredirect after restore")

            # Bind map event so we reapply override when restored
            try:
                root.bind('<Map>', _on_map)
            except Exception:
                logger.exception("Failed to bind <Map> event")
        else:
            # Normal case
            root.iconify()
    except Exception as e:
        logger.exception("Error while minimizing window: %s", e)
        # Final fallback
        try:
            root.withdraw()
        except Exception:
            pass

def close_window():
    """Close the application"""
    on_closing()

def get_video_info(filepath):
    """Get video duration, bitrate, and resolution using ffprobe"""
    try:
        # Get duration
        cmd_duration = [ffprobe_path, "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", filepath]
        result_duration = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if result_duration.returncode != 0:
            raise Exception(f"FFprobe error: {result_duration.stderr}")
        
        duration_output = result_duration.stdout.strip()
        if duration_output == 'N/A' or not duration_output:
            raise ValueError("Duration not available")
            
        duration = float(duration_output)
        
        # Get bitrate (in kilobits per second)
        cmd_bitrate = [ffprobe_path, "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "stream=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1", filepath]
        result_bitrate = subprocess.run(cmd_bitrate, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                      text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        bitrate = 0
        if result_bitrate.returncode == 0 and result_bitrate.stdout.strip():
            bitrate_output = result_bitrate.stdout.strip()
            if bitrate_output != 'N/A' and bitrate_output:
                bitrate = float(bitrate_output) / 1000  # Convert to kbps
        
        if bitrate <= 0:
            # Estimate bitrate based on file size and duration
            file_size_kb = os.path.getsize(filepath) / 1024
            bitrate = (file_size_kb * 8) / duration if duration > 0 else 5000
        
        # Get resolution
        cmd_resolution = [ffprobe_path, "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", filepath]
        result_resolution = subprocess.run(cmd_resolution, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        resolution = "1920x1080"  # Default
        if result_resolution.returncode == 0 and result_resolution.stdout.strip():
            resolution_output = result_resolution.stdout.strip()
            if resolution_output and resolution_output != 'N/A':
                resolution = resolution_output
            
        return duration, bitrate, resolution
    except Exception as e:
        messagebox.showwarning("Warning", f"Could not determine video information: {str(e)}. Using default estimates.")
        return 60.0, 5000, "1920x1080"  # Default duration, bitrate and resolution

def calculate_target_bitrate(encoder_type, original_bitrate, target_vmaf, resolution):
    """Calculate target bitrate based on VMAF target, encoder efficiency, and resolution"""
    # Base bitrate for 1080p content at VMAF 95 for different encoders
    base_bitrates = {
        "CPU (x265)": 2500,    # Most efficient
        "Intel QSV": 3000,     # Good efficiency
        "NVENC": 3500,         # Moderate efficiency
        "AMD AMF": 4000        # Least efficient of these options
    }
    
    # VMAF to bitrate adjustment factors (how much bitrate changes per VMAF point)
    vmaf_adjustment = {
        "CPU (x265)": 150,     # kbps per VMAF point
        "Intel QSV": 200,      # kbps per VMAF point
        "NVENC": 250,          # kbps per VMAF point
        "AMD AMF": 300         # kbps per VMAF point
    }
    
    # Resolution factor (higher resolutions need more bitrate)
    try:
        width, height = map(int, resolution.split('x'))
        resolution_factor = math.sqrt(width * height) / math.sqrt(1920 * 1080)  # Relative to 1080p
    except:
        resolution_factor = 1.0
    
    # Calculate base bitrate for this encoder
    base_bitrate = base_bitrates.get(encoder_type, 3000)
    
    # Adjust for VMAF target (95 is the baseline)
    vmaf_difference = target_vmaf - 95
    adjustment = vmaf_adjustment.get(encoder_type, 200) * vmaf_difference
    
    # Calculate target bitrate
    target_bitrate = (base_bitrate + adjustment) * resolution_factor
    
    # Ensure we're not increasing bitrate beyond original (unless targeting very high VMAF)
    if target_vmaf <= 98:
        target_bitrate = min(target_bitrate, original_bitrate * 1.1)  # Allow 10% increase max
    
    # Ensure reasonable minimum and maximum bitrates
    return max(800, min(20000, target_bitrate))

def get_encoder_parameters(encoder_type, target_bitrate, target_vmaf):
    """Get optimal encoder parameters based on target VMAF and bitrate"""
    params = {}
    
    if encoder_type == "CPU (x265)":
        # x265 parameters - CRF based approach with VMAF tuning
        base_crf = 23  # Base CRF for ~95 VMAF
        crf_adjustment = (98 - target_vmaf) * 1.2  # Lower CRF for higher VMAF
        
        target_crf = max(15, min(28, base_crf - crf_adjustment))
        
        params = {
            "codec": "libx265",
            "preset": "medium" if target_vmaf < 96 else "slow",
            "crf": str(target_crf),
            "tune": "ssim" if target_vmaf > 96 else "psnr",
            "profile": "main",  # Keep as main for better compatibility
            "extra_params": ["-x265-params", "log-level=error"]
        }
        
    elif encoder_type == "Intel QSV":
        # Intel QSV parameters - quality based approach
        base_quality = 23  # Base quality for ~95 VMAF
        quality_adjustment = (98 - target_vmaf) * 1.5  # Lower quality value for higher VMAF
        
        target_quality = max(18, min(30, base_quality - quality_adjustment))
        
        params = {
            "codec": "hevc_qsv",
            "preset": "veryslow" if target_vmaf > 95 else "slow",
            "global_quality": str(target_quality),
            "maxrate": f"{target_bitrate * 1.5}k",
            "bufsize": f"{target_bitrate * 2}k",
            "profile": "main",  # Keep as main for better compatibility
            "extra_params": [
                "-low_delay_brc", "1", "-extbrc", "1", "-look_ahead_depth", "100",
                "-g", "48", "-keyint_min", "48", "-sc_threshold", "0", "-b_strategy", "1",
                "-vtag", "hvc1", "-pix_fmt", "nv12"
            ]
        }
        
    elif encoder_type == "NVENC":
        # NVENC parameters - bitrate based approach with quality tuning
        preset_map = {
            "90-93": "p7",
            "94-96": "p6", 
            "97-98": "p5"
        }
        
        # Select preset based on VMAF target
        preset = "p6"  # Default
        for range_str, p_val in preset_map.items():
            low, high = map(int, range_str.split('-'))
            if low <= target_vmaf <= high:
                preset = p_val
                break
        
        params = {
            "codec": "hevc_nvenc",
            "preset": preset,
            "tune": "hq",
            "rc": "vbr",
            "b:v": f"{target_bitrate}k",
            "maxrate": f"{target_bitrate * 1.5}k",
            "bufsize": f"{target_bitrate * 2}k",
            "profile": "main",  # Keep as main for better compatibility
            "extra_params": [
                "-spatial_aq", "1", "-temporal_aq", "1",
                "-rc-lookahead", "32" if target_vmaf > 95 else "16"
            ]
        }
        
    elif encoder_type == "AMD AMF":
        # AMD AMF parameters - bitrate based approach
        quality_map = {
            "90-92": "speed",
            "93-95": "balanced",
            "96-98": "quality"
        }
        
        # Select quality based on VMAF target
        quality = "balanced"  # Default
        for range_str, q_val in quality_map.items():
            low, high = map(int, range_str.split('-'))
            if low <= target_vmaf <= high:
                quality = q_val
                break
        
        params = {
            "codec": "hevc_amf",
            "quality": quality,
            "rc": "vbr_peak",
            "b:v": f"{target_bitrate}k",
            "maxrate": f"{target_bitrate * 1.5}k",
            "bufsize": f"{target_bitrate * 2}k",
            "profile": "main",  # Keep as main for better compatibility
            "extra_params": [
                "-header_insertion_mode", "idr",
                "-preanalysis", "1" if target_vmaf > 94 else "0"
            ]
        }
    
    return params

def estimate_filesize(duration_sec, bitrate_kbps):
    """Estimate output file size based on target bitrate"""
    kilobits = bitrate_kbps * duration_sec
    megabytes = kilobits / (8 * 1024)  # Convert kilobits to megabytes
    return round(megabytes, 1)

def parse_ffmpeg_time_output(line):
    """Parse time from ffmpeg output line"""
    time_match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
    if time_match:
        hours, minutes, seconds = map(float, time_match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return None

def update_progress(progress, elapsed_time=None, remaining_time=None, current_speed=None):
    """Update progress bar and status labels"""
    progress_var.set(int(progress * 100))
    progress_bar.update()
    
    if elapsed_time is not None:
        # Format time for display
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        remaining_str = time.strftime('%H:%M:%S', time.gmtime(remaining_time)) if remaining_time else "--:--:--"
        
        speed_text = f" | Speed: {current_speed:.1f}x" if current_speed else ""
        time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}{speed_text}")

def update_vmaf_slider_label(value):
    """Update the VMAF slider label with the current value"""
    vmaf_slider_label.config(text=f"Target VMAF: {value}")

def update_estimates():
    """Update file size and bitrate estimates based on current settings"""
    if not current_file or not os.path.exists(current_file):
        return
    
    try:
        # Get current values
        encoder_type = encoder_var.get()
        target_vmaf = vmaf_var.get()
        
        # Get file info
        duration, original_bitrate, resolution = get_video_info(current_file)
        
        # Calculate target bitrate
        target_bitrate = calculate_target_bitrate(encoder_type, original_bitrate, target_vmaf, resolution)
        
        # Estimate file size
        estimated_size = estimate_filesize(duration, target_bitrate)
        file_size_mb = round(os.path.getsize(current_file) / (1024 * 1024), 1)
        
        # Update UI
        compression_ratio = round((file_size_mb - estimated_size) / file_size_mb * 100, 1) if file_size_mb > 0 else 0
        size_label.config(text=f"Original: {file_size_mb} MB → Estimated: {estimated_size} MB ({compression_ratio}% reduction)")
        bitrate_label.config(text=f"Original: {original_bitrate/1000:.1f} Mbps → Target: {target_bitrate/1000:.1f} Mbps")
        
    except Exception as e:
        logger.exception("Error updating estimates: %s", e)

def encode_video(filepath):
    global process, cancel_requested, output_file, current_file, current_encoding_index, batch_results, is_batch_processing
    cancel_requested = False
    current_file = filepath
    # Update queue status to Processing
    try:
        def _upd():
            update_queue_status(filepath, "Processing…")
        # If called from worker thread, schedule on Tk loop
        try:
            root.after(0, _upd)
        except Exception:
            _upd()
    except Exception:
        pass
    
    try:
        # Validate input file
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get current settings
        encoder_type = encoder_var.get()
        target_vmaf = vmaf_var.get()
        
        # Get file info
        file_size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)
        duration, original_bitrate, resolution = get_video_info(filepath)
        
        # Calculate target bitrate based on VMAF target
        target_bitrate = calculate_target_bitrate(encoder_type, original_bitrate, target_vmaf, resolution)
        
        # Get encoder parameters for this VMAF target
        encoder_params = get_encoder_parameters(encoder_type, target_bitrate, target_vmaf)
        
        # Estimate file size
        estimated_size = estimate_filesize(duration, target_bitrate)
        
        # Update UI with file info
        file_info_label.config(text=f"Input: {os.path.basename(filepath)} ({file_size_mb} MB, {resolution})")
        compression_ratio = round((file_size_mb - estimated_size) / file_size_mb * 100, 1) if file_size_mb > 0 else 0
        size_label.config(text=f"Original: {file_size_mb} MB → Estimated: {estimated_size} MB ({compression_ratio}% reduction)")
        bitrate_label.config(text=f"Original: {original_bitrate/1000:.1f} Mbps → Target: {target_bitrate/1000:.1f} Mbps")
        vmaf_label.config(text=f"Target VMAF: {target_vmaf} | Encoder: {encoder_type}")
        
        # Prepare output filename
        base, ext = os.path.splitext(filepath)
        output_file = base + f"_encoded_vmaf{target_vmaf}" + ext
        
        # Check if output file already exists
        if os.path.exists(output_file):
            if not messagebox.askyesno("File exists", 
                                      f"The file {os.path.basename(output_file)} already exists. Overwrite?"):
                status_label.config(text="Ready to encode")
                return
        
        # Build ffmpeg command based on selected encoder and VMAF target
        cmd = [ffmpeg_path, "-y", "-i", filepath]
        
        # Add video encoding options
        cmd.extend(["-c:v", encoder_params["codec"]])
        
        # Add encoder-specific parameters
        if encoder_type == "CPU (x265)":
            cmd.extend(["-preset", encoder_params["preset"]])
            cmd.extend(["-crf", encoder_params["crf"]])
            cmd.extend(["-tune", encoder_params["tune"]])
            cmd.extend(["-profile:v", encoder_params["profile"]])
            cmd.extend(encoder_params["extra_params"])
            
        elif encoder_type == "Intel QSV":
            cmd.extend(["-preset", encoder_params["preset"]])
            cmd.extend(["-global_quality", encoder_params["global_quality"]])
            if "maxrate" in encoder_params:
                cmd.extend(["-maxrate", encoder_params["maxrate"]])
            if "bufsize" in encoder_params:
                cmd.extend(["-bufsize", encoder_params["bufsize"]])
            cmd.extend(["-profile:v", encoder_params["profile"]])
            cmd.extend(encoder_params["extra_params"])
            
        elif encoder_type == "NVENC":
            cmd.extend(["-preset", encoder_params["preset"]])
            cmd.extend(["-tune", encoder_params["tune"]])
            cmd.extend(["-rc", encoder_params["rc"]])
            cmd.extend(["-b:v", encoder_params["b:v"]])
            if "maxrate" in encoder_params:
                cmd.extend(["-maxrate", encoder_params["maxrate"]])
            if "bufsize" in encoder_params:
                cmd.extend(["-bufsize", encoder_params["bufsize"]])
            cmd.extend(["-profile:v", encoder_params["profile"]])
            cmd.extend(encoder_params["extra_params"])
            
        elif encoder_type == "AMD AMF":
            cmd.extend(["-quality", encoder_params["quality"]])
            cmd.extend(["-rc", encoder_params["rc"]])
            cmd.extend(["-b:v", encoder_params["b:v"]])
            if "maxrate" in encoder_params:
                cmd.extend(["-maxrate", encoder_params["maxrate"]])
            if "bufsize" in encoder_params:
                cmd.extend(["-bufsize", encoder_params["bufsize"]])
            cmd.extend(["-profile:v", encoder_params["profile"]])
            cmd.extend(encoder_params["extra_params"])
        
        # Audio options (copy original audio)
        cmd.extend(["-c:a", "copy", output_file])
        
        # Start encoding process
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        # Log the command for debugging
        logger.info("FFmpeg command: %s", " ".join(shlex.quote(c) for c in cmd))

        # Use Popen with proper error handling
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   stdin=subprocess.PIPE, text=True, startupinfo=startupinfo,
                                   creationflags=subprocess.CREATE_NO_WINDOW, bufsize=1)
        
        start_time = time.time()
        last_progress_update = start_time
        speed_samples = []
        
        # Monitor process
        for line in process.stdout:
            if cancel_requested:
                break
                
            line = line.strip()
            current_time = parse_ffmpeg_time_output(line)
            
            if current_time is not None and duration > 0:
                # Calculate progress
                progress = min(current_time / duration, 0.99)  # Cap at 99% until done
                
                # Calculate elapsed time
                elapsed = time.time() - start_time
                
                # Calculate encoding speed (real-time factor)
                if current_time > 0 and elapsed > 0:
                    current_speed = current_time / elapsed
                    speed_samples.append(current_speed)
                    if len(speed_samples) > 10:  # Keep last 10 samples
                        speed_samples.pop(0)
                    avg_speed = sum(speed_samples) / len(speed_samples) if speed_samples else current_speed
                    
                    # Calculate remaining time
                    if progress > 0.05:  # Wait until we have some progress
                        remaining = (elapsed / progress) - elapsed
                    else:
                        remaining = duration - progress
                    
                    # Update progress every 0.2 seconds to avoid UI lag
                    if time.time() - last_progress_update > 0.2:
                        update_progress(progress, elapsed, remaining, avg_speed)
                        last_progress_update = time.time()
            
            # Check for completion
            if process.poll() is not None:
                break
        
        if cancel_requested:
            # Use os.kill to properly terminate the process
            if process and process.poll() is None:
                try:
                    # Try to terminate gracefully first
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate
                        process.kill()
                except:
                    logger.exception("Error terminating ffmpeg process")
            
            # Clean up partial output file
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    logger.exception("Error removing partial output file %s", output_file)
            
            status_label.config(text="❌ Cancelled")
            try:
                update_queue_status(filepath, "Cancelled")
            except Exception:
                pass
            return
        
        # Process completed
        return_code = process.wait()
        
        # Handle the specific error code 4294967268 (which is -20 in signed 32-bit)
        if return_code == 4294967268:
            # This is actually a negative return code that got converted to unsigned
            # It typically means the process was terminated
            if cancel_requested:
                status_label.config(text="❌ Cancelled")
                return
            else:
                raise Exception("Encoding process was terminated unexpectedly")
        
        if return_code == 0:
            # Encoding successful
            output_size = round(os.path.getsize(output_file) / (1024 * 1024), 1)
            oversized = output_size >= file_size_mb
            if oversized:
                # Delete oversized output; it's not needed
                try:
                    os.remove(output_file)
                except Exception:
                    logger.exception("Failed to remove oversized output %s", output_file)
                status_label.config(text="⚠️ Output larger than original; discarded")
                output_size_effective = file_size_mb
                compression_ratio = 0.0
                size_label.config(text=f"Output discarded (larger than original: {output_size} ≥ {file_size_mb} MB)")
                try:
                    update_queue_status(filepath, "Discarded (larger)")
                except Exception:
                    pass
            else:
                compression_ratio = round((file_size_mb - output_size) / file_size_mb * 100, 1)
                status_label.config(text="✅ Encoding complete!")
                size_label.config(text=f"Output: {output_size} MB ({compression_ratio}% smaller)")
                try:
                    update_queue_status(filepath, f"Encoded ({compression_ratio}% smaller)")
                except Exception:
                    pass
            
            total_time = time.time() - start_time
            update_progress(1.0, total_time, 0, duration/total_time if total_time > 0 else 1)
            
            # Store result for batch summary
            batch_results.append({
                'filename': os.path.basename(filepath),
                'original_size': file_size_mb,
                'output_size': output_size if not oversized else file_size_mb,
                'compression_ratio': compression_ratio,
                'time_taken': total_time
            })
            
            # Delete original file if checkbox is checked and output is smaller
            if delete_original_var.get() and (not oversized) and output_size < file_size_mb:
                try:
                    os.remove(filepath)
                    status_label.config(text=status_label.cget("text") + " | Original deleted")
                except Exception as e:
                            logger.warning("Could not delete original file %s: %s", filepath, e)
            
            # Do not show a modal success message here — for batch processing
            # we want to avoid interrupting the queue. A final summary or a
            # single-file success dialog will be shown after the queue finishes.
        else:
            # Encoding failed
            error_msg = f"Encoding failed with error code {return_code}"
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
            raise Exception(error_msg)
            
    except Exception as e:
        error_msg = f"Error during encoding: {str(e)}"
        status_label.config(text="❌ Error")
        logger.exception("Encoding error: %s", e)
        messagebox.showerror("Encoding Error", error_msg)
        try:
            update_queue_status(filepath, "Error")
        except Exception:
            pass
    finally:
        # Clean up UI
        try:
            cancel_btn.pack_forget()
        except Exception:
            pass
        if not cancel_requested and status_label.cget("text") != "✅ Encoding complete!":
            status_label.config(text="Ready to encode")
        
        # Process next item in queue
        process_next_in_queue()

def process_next_in_queue():
    """Process the next item in the encoding queue"""
    global encoding_thread, current_encoding_index, is_batch_processing, batch_results
    
    if encoding_queue:
        # Remove the completed item from the internal queue only (keep row to show outcome)
        if current_encoding_index >= 0:
            try:
                encoding_queue.popleft()
            except IndexError:
                pass
        
        # Process next item if available
        if encoding_queue:
            current_encoding_index = 0
            filepath = encoding_queue[0]
            # select and update status in the UI
            try:
                item_id = queue_items_map.get(filepath)
                if item_id:
                    queue_tree.selection_set(item_id)
                    queue_tree.see(item_id)
                update_queue_status(filepath, "Processing…")
            except Exception:
                pass
            
            if cancel_requested:
                status_label.config(text=f"✅ Cancelled previous. ▶ Starting next: {os.path.basename(filepath)}")
            else:
                status_label.config(text="⏳ Preparing...")
            start_btn.config(state=tk.DISABLED)
            try:
                start_btn.pack_forget()
            except Exception:
                pass
            try:
                cancel_btn.pack()
            except Exception:
                pass
            
            encoding_thread = threading.Thread(target=encode_video, args=(filepath,), daemon=True)
            encoding_thread.start()
        else:
            current_encoding_index = -1
            is_batch_processing = False
            
            # Show batch summary if we had multiple files
            if len(batch_results) > 1:
                show_batch_summary()
            elif len(batch_results) == 1:
                # Single-file encoding completed; show a success dialog
                r = batch_results[0]
                try:
                    messagebox.showinfo("Success",
                                        f"Encoding completed successfully!\n\n"
                                        f"Original: {r['original_size']:.1f} MB\n"
                                        f"Encoded: {r['output_size']:.1f} MB ({r['compression_ratio']:.1f}% reduction)")
                except Exception:
                    logger.exception("Could not show single-file success dialog")
            
            batch_results = []
            status_label.config(text="Ready to encode")
            start_btn.config(state=tk.NORMAL)
            try:
                cancel_btn.pack_forget()
            except Exception:
                pass
            try:
                start_btn.pack()
            except Exception:
                pass
    else:
        current_encoding_index = -1
        is_batch_processing = False
        batch_results = []
        status_label.config(text="Ready to encode")
        start_btn.config(state=tk.NORMAL)
        try:
            cancel_btn.pack_forget()
        except Exception:
            pass
        try:
            start_btn.pack()
        except Exception:
            pass

def show_batch_summary():
    """Show summary of batch encoding results"""
    total_original = sum(item['original_size'] for item in batch_results)
    total_output = sum(item['output_size'] for item in batch_results)
    total_saved = total_original - total_output
    total_time = sum(item['time_taken'] for item in batch_results)
    avg_compression = sum(item['compression_ratio'] for item in batch_results) / len(batch_results)
    
    summary = f"Batch encoding completed!\n\n"
    summary += f"Files processed: {len(batch_results)}\n"
    summary += f"Total original size: {total_original:.1f} MB\n"
    summary += f"Total encoded size: {total_output:.1f} MB\n"
    summary += f"Total space saved: {total_saved:.1f} MB\n"
    summary += f"Average compression: {avg_compression:.1f}%\n"
    summary += f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}"
    
    messagebox.showinfo("Batch Complete", summary)

def add_to_queue(filepath):
    """Add a file to the encoding queue"""
    # Validate file type
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v')
    if not filepath.lower().endswith(video_extensions):
        messagebox.showerror("Invalid File", "Please drop a video file (MP4, AVI, MOV, MKV, etc.)")
        return False
    
    # Check if file is already in queue
    if filepath in encoding_queue:
        messagebox.showinfo("Info", "This file is already in the queue.")
        return False
    
    # Add to queue and UI
    encoding_queue.append(filepath)
    try:
        basename = os.path.basename(filepath)
        item_id = queue_tree.insert("", "end", values=(basename, "Queued"))
        queue_items_map[filepath] = item_id
        item_to_path[item_id] = filepath
    except Exception:
        logger.exception("Failed to add item to queue view")
    
    # Update file info if this is the first file
    if len(encoding_queue) == 1 and not (encoding_thread and encoding_thread.is_alive()):
        try:
            duration, original_bitrate, resolution = get_video_info(filepath)
            file_size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)
            file_info_label.config(text=f"Input: {os.path.basename(filepath)} ({file_size_mb} MB, {resolution})")
            update_estimates()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file information: {str(e)}")
    # Ensure start button state is updated (for older Python/Tkinter event ordering)
    _maybe_enable_start()
    return True

def remove_from_queue():
    """Remove selected item from queue"""
    selection = queue_tree.selection()
    if selection:
        item_id = selection[0]
        filepath = item_to_path.get(item_id)
        try:
            queue_tree.delete(item_id)
        except Exception:
            pass
        if filepath in queue_items_map:
            queue_items_map.pop(filepath, None)
        if item_id in item_to_path:
            item_to_path.pop(item_id, None)

        # Remove filepath from internal queue if present
        try:
            if filepath in encoding_queue:
                temp = list(encoding_queue)
                try:
                    idx = temp.index(filepath)
                    temp.pop(idx)
                except ValueError:
                    pass
                encoding_queue.clear()
                encoding_queue.extend(temp)
                # If removed currently encoding file, cancel and move on
                if current_encoding_index == 0 and idx == 0:
                    cancel_encoding()
                    process_next_in_queue()
                elif current_encoding_index > 0 and idx < current_encoding_index:
                    current_encoding_index -= 1
        except Exception:
            logger.exception("Error removing file from internal queue")
    _maybe_enable_start()

def drop_handler(event):
    """Handle file drop event"""
    global is_batch_processing
    
    filepaths = event.data
    if isinstance(filepaths, str):
        # Handle single file or multiple files wrapped in curly braces
        if filepaths.startswith('{') and filepaths.endswith('}'):
            filepaths = filepaths[1:-1].split('} {')
        else:
            filepaths = [filepaths]
    
    added_files = 0
    for filepath in filepaths:
        filepath = filepath.strip()
        if filepath:
            if add_to_queue(filepath):
                added_files += 1
    
    # Set batch processing flag if multiple files
    if added_files > 1:
        is_batch_processing = True
    
    # If not currently encoding, start processing the queue
    if not (encoding_thread and encoding_thread.is_alive()) and encoding_queue:
        process_next_in_queue()

def start_encoding():
    """Start the encoding process"""
    global is_batch_processing
    
    if not encoding_queue:
        messagebox.showerror("Error", "No video files in queue. Please drop video files first.")
        return
    
    # Set batch processing flag if multiple files
    if len(encoding_queue) > 1:
        is_batch_processing = True
    
    process_next_in_queue()

def cancel_encoding():
    """Cancel the current encoding process"""
    global cancel_requested
    if messagebox.askyesno("Cancel", "Are you sure you want to cancel the encoding process?"):
        cancel_requested = True
        status_label.config(text="⏳ Cancelling...")

def on_closing():
    """Handle application closing"""
    global cancel_requested, process
    
    if process and process.poll() is None:
        if messagebox.askyesno("Quit", "Encoding is in progress. Are you sure you want to quit?"):
            cancel_requested = True
            # Use os.kill to properly terminate the process
            try:
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
            except:
                pass
            root.destroy()
    else:
        root.destroy()

def on_encoder_change(*args):
    """Handle encoder selection change"""
    if current_file:
        update_estimates()

def on_vmaf_change(*args):
    """Handle VMAF slider change"""
    update_vmaf_slider_label(vmaf_var.get())
    if current_file:
        update_estimates()

# Create main window without title bar
if USE_TKINTER_DND and TkinterDnD is not None:
    try:
        root = TkinterDnD.Tk()
    except Exception:
        logger.exception("Failed to initialize TkinterDnD, falling back to standard Tk()")
        root = tk.Tk()
        USE_TKINTER_DND = False
else:
    root = tk.Tk()
root.geometry("980x550")
root.title("VMAF-Based HEVC Encoder")
root.configure(bg="#222")
root.overrideredirect(True)  # Remove title bar
root.resizable(False, False)  # Make window non-resizable
root.protocol("WM_DELETE_WINDOW", on_closing)

# Verify tkdnd package is available at runtime when using TkinterDnD
if USE_TKINTER_DND:
    try:
        ver = root.tk.eval('package require tkdnd')
        logger.info("tkdnd package loaded (version %s)", ver)
    except Exception:
        logger.exception("tkdnd package not available at runtime; disabling drag-and-drop")
        USE_TKINTER_DND = False

def _try_init_tkdnd_paths():
    """Attempt to locate and load tkdnd when frozen or paths are non-standard.

    Tries TKDND_LIBRARY env var and auto_path to point to a bundled 'tkdnd2.9' dir
    located either next to the executable (resource_path) or inside the
    tkinterdnd2 package folder.
    """
    global USE_TKINTER_DND
    if not USE_TKINTER_DND:
        return
    try:
        import tkinterdnd2 as tkdnd
        # If already loaded, nothing to do
        try:
            ver = root.tk.eval('package provide tkdnd')
            if ver:
                logger.info("tkdnd already provided (version %s)", ver)
                return
        except Exception:
            pass

        candidates = []
        try:
            candidates.append(resource_path('tkdnd2.9'))
        except Exception:
            pass
        try:
            pkg_dir = os.path.dirname(tkdnd.__file__)
            candidates.append(os.path.join(pkg_dir, 'tkdnd2.9'))
        except Exception:
            pass

        for d in candidates:
            if d and os.path.isdir(d):
                try:
                    os.environ['TKDND_LIBRARY'] = d
                    ver = root.tk.eval('package require tkdnd')
                    logger.info("tkdnd loaded via TKDND_LIBRARY=%s (version %s)", d, ver)
                    return
                except Exception:
                    # try adding to auto_path
                    try:
                        root.tk.eval(f'lappend auto_path "{d}"')
                        ver = root.tk.eval('package require tkdnd')
                        logger.info("tkdnd loaded via auto_path %s (version %s)", d, ver)
                        return
                    except Exception:
                        continue
        logger.error("Could not initialize tkdnd (tkdnd2.9 not found or load failed); disabling DnD")
        USE_TKINTER_DND = False
    except Exception:
        logger.exception("Error while trying to initialize tkdnd paths")
        USE_TKINTER_DND = False

# Center window on screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - 980) // 2
y = (screen_height - 550) // 2
root.geometry(f"980x550+{x}+{y}")

# Configure styles
style = ttk.Style()
style.theme_use('clam')
style.configure("TProgressbar", thickness=20, troughcolor='#333', background='#4CAF50')
style.configure("TCombobox", fieldbackground="#333", background="#333", foreground="#eee", 
                selectbackground="#4CAF50", selectforeground="#fff")
style.configure("Horizontal.TScale", troughcolor="#333", background="#4CAF50")
style.configure("Listbox", background="#333", foreground="#eee", selectbackground="#4CAF50")

# Custom title bar
title_bar = tk.Frame(root, bg="#2c2c2c", relief="raised", bd=0, height=30)
title_bar.pack(fill="x", side="top")
title_bar.pack_propagate(False)

# Title bar text
title_label = tk.Label(title_bar, text="VMAF-Based HEVC Encoder", bg="#2c2c2c", fg="#eee", 
                       font=("Arial", 10, "bold"))
title_label.pack(side="left", padx=10)

# Window controls
controls_frame = tk.Frame(title_bar, bg="#2c2c2c")
controls_frame.pack(side="right", padx=5)

minimize_btn = tk.Button(controls_frame, text="─", bg="#2c2c2c", fg="#eee", 
                         font=("Arial", 12), bd=0, command=minimize_window,
                         activebackground="#444", activeforeground="#eee")
minimize_btn.pack(side="left", padx=2)

close_btn = tk.Button(controls_frame, text="×", bg="#2c2c2c", fg="#eee", 
                      font=("Arial", 14), bd=0, command=close_window,
                      activebackground="#e74c3c", activeforeground="#eee")
close_btn.pack(side="left", padx=2)

# Make title bar draggable
title_bar.bind("<Button-1>", start_drag)
title_bar.bind("<ButtonRelease-1>", stop_drag)
title_bar.bind("<B1-Motion>", do_drag)
title_label.bind("<Button-1>", start_drag)
title_label.bind("<ButtonRelease-1>", stop_drag)
title_label.bind("<B1-Motion>", do_drag)

# Main content - split into two frames
main_frame = tk.Frame(root, bg="#222")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Left frame for controls
left_frame = tk.Frame(main_frame, bg="#222", width=580)
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
left_frame.pack_propagate(False)

# Right frame for queue
right_frame = tk.Frame(main_frame, bg="#222", width=320)
right_frame.pack(side="right", fill="y", padx=(10, 0))
right_frame.pack_propagate(False)

# Queue label
queue_label = tk.Label(right_frame, text="Encoding Queue", bg="#222", fg="#eee", font=("Arial", 10, "bold"))
queue_label.pack(pady=(0, 5))

# Queue with status (Treeview) and scrollbar
queue_frame = tk.Frame(right_frame, bg="#222")
queue_frame.pack(fill="both", expand=True)

queue_scrollbar = tk.Scrollbar(queue_frame)
queue_scrollbar.pack(side="right", fill="y")

queue_tree = ttk.Treeview(queue_frame, columns=("filename", "status"), show="headings", selectmode="browse")
queue_tree.heading("filename", text="File")
queue_tree.heading("status", text="Outcome")
queue_tree.column("filename", anchor="w", width=200)
queue_tree.column("status", anchor="w", width=160)
queue_tree.pack(side="left", fill="both", expand=True)
queue_tree.configure(yscrollcommand=queue_scrollbar.set)
queue_scrollbar.config(command=queue_tree.yview)

# Remove from queue button
remove_btn = tk.Button(right_frame, text="Remove Selected", command=remove_from_queue,
                      bg="#d32f2f", fg="white", font=("Arial", 9), relief="flat",
                      activebackground="#b71c1c", activeforeground="white")
remove_btn.pack(pady=5)

# Encoder selection
encoder_frame = tk.Frame(left_frame, bg="#222")
encoder_frame.pack(pady=(0, 10))

enc_label = tk.Label(encoder_frame, text="Encoder:", bg="#222", fg="#eee", font=("Arial", 10))
enc_label.pack(side=tk.LEFT, padx=(0, 5))

# Detect available encoders and set default to CPU
available_encoders = detect_available_encoders()
if "CPU (x265)" not in available_encoders:
    available_encoders.insert(0, "CPU (x265)")

encoder_var = tk.StringVar(value="CPU (x265)")

# If only CPU is available, show a static label instead of a dropdown
if len(available_encoders) == 1:
    encoder_static = tk.Label(encoder_frame, text="CPU (x265)", bg="#222", fg="#aaa", font=("Arial", 10))
    encoder_static.pack(side=tk.LEFT)
    encoder_dropdown = None
else:
    encoder_dropdown = ttk.Combobox(encoder_frame, textvariable=encoder_var,
                                   values=available_encoders,
                                   state="readonly", width=16)
    encoder_dropdown.pack(side=tk.LEFT)
    # Use trace_add for Tcl 9+, fallback to trace for older versions
    try:
        encoder_var.trace_add('write', on_encoder_change)
    except Exception:
        encoder_var.trace('w', on_encoder_change)

# VMAF selection
vmaf_frame = tk.Frame(left_frame, bg="#222")
vmaf_frame.pack(pady=(10, 5))

tk.Label(vmaf_frame, text="Quality:", bg="#222", fg="#eee", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))

vmaf_var = tk.IntVar(value=95)
vmaf_slider = ttk.Scale(vmaf_frame, from_=90, to=98, variable=vmaf_var, 
                       orient="horizontal", length=200)
vmaf_slider.pack(side=tk.LEFT, padx=(0, 10))
# Use trace_add for Tcl 9+, fallback to trace for older versions
try:
    vmaf_var.trace_add('write', on_vmaf_change)
except Exception:
    vmaf_var.trace('w', on_vmaf_change)

vmaf_slider_label = tk.Label(vmaf_frame, text="Target VMAF: 95", bg="#222", fg="#4CAF50", 
                            font=("Arial", 10, "bold"))
vmaf_slider_label.pack(side=tk.LEFT)

# Delete original checkbox
delete_original_var = tk.BooleanVar(value=False)
delete_original_cb = tk.Checkbutton(left_frame, text="Delete original if smaller", 
                                   variable=delete_original_var, bg="#222", fg="#eee",
                                   selectcolor="#333", activebackground="#222", 
                                   activeforeground="#eee")
delete_original_cb.pack(pady=5)

# Drop area
drop_frame = tk.Frame(left_frame, bg="#333", relief="raised", bd=2)
drop_frame.pack(pady=10, fill="x", ipady=30)

status_label = tk.Label(drop_frame, text="Drop your video(s) here or use Add Files…", bg="#333", fg="#eee", 
                        font=("Arial", 12), relief="flat")
status_label.pack(expand=True)

if USE_TKINTER_DND and DND_FILES is not None:
    # In frozen apps, ensure tkdnd paths are initialized
    _try_init_tkdnd_paths()
    # Prefer binding on the visible drop area; also bind root as fallback (tkinterdnd2 style)
    dnd_ok = False
    for widget in (drop_frame, root):
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind('<<Drop>>', drop_handler)
            dnd_ok = True
        except Exception:
            continue
    if not dnd_ok:
        logger.exception("DnD binding failed on both drop_frame and root; drag-and-drop disabled")
        USE_TKINTER_DND = False
        try:
            status_label.config(text="Add Files… (drag & drop unavailable)")
        except Exception:
            pass
else:
    # No drag & drop available; users can use OS file dialog via the Add Files button
    logger.info("tkinterdnd2 not available; drag-and-drop disabled")

# Fallback/optional: Add Files button (works regardless of DnD availability)
def add_files_dialog():
    try:
        paths = filedialog.askopenfilenames(title="Select video files",
                                            filetypes=[
                                                ("Video files", 
                                                 ".mp4 .avi .mov .mkv .flv .wmv .webm .m4v"),
                                                ("All files", "*.*")
                                            ])
        if not paths:
            return
        added = 0
        for p in paths:
            try:
                if add_to_queue(p):
                    added += 1
            except Exception:
                logger.exception("Failed adding file from dialog: %s", p)
        if added:
            try:
                status_label.config(text=f"Added {added} file(s) to queue")
            except Exception:
                pass
        # Auto-start if idle
        if not (encoding_thread and encoding_thread.is_alive()) and encoding_queue:
            process_next_in_queue()
    except Exception:
        logger.exception("Error in file open dialog")

add_files_btn = tk.Button(left_frame, text="Add Files…", command=add_files_dialog,
                          bg="#4a4a4a", fg="#eee", font=("Arial", 9), relief="flat",
                          activebackground="#5a5a5a", activeforeground="#fff")
add_files_btn.pack(pady=(5, 0))

def _maybe_enable_start():
    # Enable start button when there's at least one file in queue
    if encoding_queue:
        start_btn.config(state=tk.NORMAL)
    else:
        start_btn.config(state=tk.DISABLED)

def update_queue_status(filepath, status_text):
    """Update the Outcome column for the given filepath in the queue tree."""
    try:
        item_id = queue_items_map.get(filepath)
        if not item_id:
            return
        vals = queue_tree.item(item_id, "values")
        filename = vals[0] if vals else os.path.basename(filepath)
        queue_tree.item(item_id, values=(filename, status_text))
    except Exception:
        logger.exception("Failed to update queue status for %s", filepath)

# File info
file_info_label = tk.Label(left_frame, text="", bg="#222", fg="#aaa", font=("Arial", 9))
file_info_label.pack(pady=(15, 5))

# Size info
size_label = tk.Label(left_frame, text="", bg="#222", fg="#aaa", font=("Arial", 10))
size_label.pack(pady=5)

# Bitrate info
bitrate_label = tk.Label(left_frame, text="", bg="#222", fg="#aaa", font=("Arial", 10))
bitrate_label.pack(pady=5)

# VMAF info
vmaf_label = tk.Label(left_frame, text="", bg="#222", fg="#4CAF50", font=("Arial", 10, "bold"))
vmaf_label.pack(pady=5)

# Progress bar
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(left_frame, variable=progress_var, maximum=100, mode='determinate')
progress_bar.pack(pady=10, fill="x")

# Time info
time_label = tk.Label(left_frame, text="Elapsed: 00:00:00 | Remaining: --:--:--", 
                      bg="#222", fg="#888", font=("Arial", 9))
time_label.pack(pady=5)

# Button row (consistent placement for Start/Cancel)
button_frame = tk.Frame(left_frame, bg="#222")
button_frame.pack(pady=5)

# Start button
start_btn = tk.Button(button_frame, text="▶ Start Encoding", command=start_encoding, 
                     bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), relief="flat",
                     activebackground="#45a049", activeforeground="white", state=tk.DISABLED)
start_btn.pack()

# Cancel button (initially hidden)
cancel_btn = tk.Button(button_frame, text="❌ Cancel", command=cancel_encoding, 
                       bg="#d32f2f", fg="white", font=("Arial", 10), relief="flat",
                       activebackground="#b71c1c", activeforeground="white")

# Add some helpful info at the bottom
help_text = tk.Label(left_frame, text="Supports: MP4, AVI, MOV, MKV, FLV, WMV, WebM, M4V\nConcept and design by amol.more@hotmail.com • Built with help from DeepSeek.ai", 
                     bg="#222", fg="#555", font=("Arial", 8))
help_text.pack(side="bottom", pady=5)

root.mainloop()
