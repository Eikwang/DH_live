@echo off
setlocal enabledelayedexpansion

REM Set your desired output directory here
set "output_dir=output"
if not exist "%output_dir%" mkdir "%output_dir%"

REM Iterate over all video files in the current directory
for %%f in (*.avi *.mkv *.mp4 *.mov *.wmv) do (
    set "input_file=%%~nf"
    set "output_file=%output_dir%\!input_file!.mp4"
    
    ffmpeg -i "%%f" -c:v hevc_nvenc -r 25 -preset slow -cq 20 ^
           -c:a aac -ar 16000 -ac 1 ^
           -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ^
           -strict experimental ^
           "!output_file!"
    
    if errorlevel 1 (
        echo Failed to process file: %%f
    ) else (
        echo Successfully processed file: %%f
    )
)

endlocal
pause