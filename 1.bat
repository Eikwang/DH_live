@echo off
:: 静音音频生成器 for Windows
:: 依赖：ffmpeg 和 ffprobe 必须在系统 PATH 中

set "VIDEO=%1.mp4"
set "OUTPUT=silent_audio.wav"

echo 正在获取视频时长...
for /f %%i in ('ffprobe -v quiet -show_entries format^=duration -of csv^=p^=0 "%VIDEO%"') do set DURATION=%%i

:: 检查是否成功获取时长
if not defined DURATION (
    echo ❌ 错误：无法获取视频时长，请检查文件是否存在或 ffprobe 是否可用。
    pause
    exit /b 1
)

echo 视频时长: %DURATION% 秒
echo.

echo 正在生成静音音频...
ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 ^
       -t %DURATION% ^
       -c:a pcm_s16le ^
       "%OUTPUT%"

if %errorlevel% equ 0 (
    echo ✅ 静音音频已生成: %OUTPUT%
) else (
    echo ❌ FFmpeg 执行失败，请检查是否安装 ffmpeg 并加入系统 PATH。
)

echo.
pause