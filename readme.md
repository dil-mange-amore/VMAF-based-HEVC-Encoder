# VMAF-Based HEVC Encoder

This is a graphical user interface (GUI) application for Windows that allows users to encode videos to the HEVC (H.265) format, targeting a specific perceptual quality level measured by Netflix's VMAF (Video Multi-Method Assessment Fusion) score.

Instead of guessing a bitrate, you can choose a target quality (e.g., VMAF 95 for excellent quality that is visually lossless to most), and the application will determine the optimal bitrate to achieve that quality, often resulting in significant file size savings with minimal quality loss.

## Features

- **Simple Drag-and-Drop Interface**: Easily add one or more video files to the encoding queue.
- **VMAF-Targeted Encoding**: Select a target VMAF score (from 90 to 98) to define the desired output quality.
- **Multiple Encoder Support**: Automatically detects and utilizes available hardware and software encoders:
    - **CPU (x265)**: High-quality software encoding.
    - **NVENC**: NVIDIA's hardware encoder for fast performance.
    - **Intel QSV**: Intel's Quick Sync Video for hardware acceleration.
    - **AMD AMF**: AMD's Advanced Media Framework for hardware encoding.
- **Batch Processing**: Encode multiple files in a queue.
- **Real-time Progress**: Monitor encoding progress with elapsed time, estimated remaining time, and speed.
- **File Size Estimation**: Get an estimate of the output file size before starting the encode.
- **Smart File Handling**:
    - Option to automatically delete the original file if the new file is smaller.
    - Automatically discards encodes that result in a larger file than the original.
- **Audio Passthrough**: Copies the original audio track without re-encoding to preserve quality.

## How It Works

1.  **Drop Files**: Drag and drop your video files (e.g., MP4, MKV, MOV) onto the application window.
2.  **Set Quality**: Use the slider to choose a target VMAF score. A higher score means better quality and a larger file size. VMAF 95 is a good starting point for high-quality encodes.
3.  **Choose Encoder**: Select the desired encoder. Hardware encoders (NVENC, QSV, AMF) are much faster, while the CPU (x265) encoder is slower but can offer slightly better compression.
4.  **Start Encoding**: The application uses `ffmpeg` to re-encode the video with parameters optimized to meet your VMAF target.

The application intelligently calculates the required bitrate based on the source video's resolution, the selected encoder's efficiency, and the target VMAF score.

## Requirements

- Windows OS.
- **FFmpeg**: The application requires `ffmpeg.exe` and `ffprobe.exe` to function. It will first look for them in the same folder as the application. If not found, it will search your system's PATH. You can download FFmpeg from [here](https://ffmpeg.org/download.html).

## Building from Source

If you want to build the application yourself:

1.  **Clone the repository.**
2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt 
    ```
    *(Note: You may need to create a `requirements.txt` file with `tkinterdnd2` and `pyinstaller`)*
3.  **Bundle with PyInstaller**:
    Use the provided `VMAFVideoEncoder.spec` file to build the executable:
    ```bash
    pyinstaller VMAFVideoEncoder.spec
    ```
    This will create a single-file executable in the `dist` folder.
