@echo off
echo Building VMAF Video Encoder...
echo.

REM Check if all required files exist
set files=ffmpeg.exe ffprobe.exe avcodec-61.dll avdevice-61.dll avfilter-10.dll avformat-61.dll avutil-59.dll postproc-58.dll swresample-5.dll swscale-8.dll

for %%f in (%files%) do (
    if not exist "%%f" (
        echo Error: Missing required file: %%f
        pause
        exit /b 1
    )
)

echo All required files found.
echo.

REM Run PyInstaller using the spec file (bundles tkinterdnd2 data for drag & drop)
pyinstaller VMAFVideoEncoder.spec

echo.
echo Build complete! Check the 'dist' folder for the executable.
pause