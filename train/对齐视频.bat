@echo off
setlocal enabledelayedexpansion

REM 设置输入和输出目录
set INPUT_DIR=D:\AI\DH_live\train\data
set OUTPUT_DIR=output

REM 创建输出目录（如果不存在）
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 遍历输入目录中的所有视频文件
for %%f in ("%INPUT_DIR%\*.mp4" "%INPUT_DIR%\*.avi" "%INPUT_DIR%\*.mkv" "%INPUT_DIR%\*.mov" "%INPUT_DIR%\*.wmv") do (
    REM 获取文件名（不含扩展名）
    set FILENAME=%%~nf
    
    REM 构建输出文件路径（强制输出为.mp4）
    set OUTPUT_FILE=%OUTPUT_DIR%\!FILENAME!.mp4

    REM 执行FFmpeg合并处理
    ffmpeg -i "%%f" ^
        -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2" ^
        -c:v hevc_nvenc -r 25 -preset slow -cq 20 ^
        -c:a aac -ar 16000 -ac 1 ^
        -strict experimental ^
        -movflags +faststart -shortest ^
        "!OUTPUT_FILE!"

    REM 检查FFmpeg执行结果
    if errorlevel 1 (
        echo 错误：处理失败 - %%f
    ) else (
        echo 成功：已处理 - %%f
    )
)

echo 所有文件处理完成。
pause