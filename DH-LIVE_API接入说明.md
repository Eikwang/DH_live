# DH-LIVE API接入说明

## 概述

DH-LIVE是一个实时数字人渲染系统，提供音频驱动的数字人口型同步和实时视频输出功能。本文档详细说明了如何通过API接口与DH-LIVE系统进行集成。

## 系统特性

- **实时渲染**：支持实时音频驱动的数字人渲染
- **虚拟摄像头输出**：渲染结果直接输出到虚拟摄像头设备
- **多种音频输入**：支持文件上传和Base64编码音频数据
- **一键启动**：系统启动时自动初始化所有组件
- **灵活配置**：支持自定义音频模型、渲染模型和角色

## 快速开始

### 1. 启动DH-LIVE系统

#### 方法一：使用批处理文件（推荐）
```bash
# 使用默认参数启动
启动DH-LIVE带参数.bat

# 或指定自定义参数
启动DH-LIVE带参数.bat --character your_character --port 8051
```

#### 方法二：直接运行Python脚本
```bash
# 激活虚拟环境
conda activate dh_live

# 启动系统
python dh_live_realtime.py --character dw --port 8051
```

### 2. 验证系统状态

启动后，系统会显示以下信息：
```
启动DH-LIVE系统...
音频模型: checkpoint/audio.pkl
渲染模型: checkpoint/15000.pth
角色: dw
服务地址: localhost:8051
DH-LIVE系统已自动启动，等待音频输入...
API服务启动在: http://localhost:8051
可用的API端点:
  POST /receive_audio - 接收音频文件
  POST /receive_audio_base64 - 接收Base64编码音频
  GET /status - 获取系统状态
```

## API接口详细说明

### 基础信息

- **服务地址**：`http://localhost:8051`（默认）
- **协议**：HTTP/HTTPS
- **数据格式**：JSON
- **支持的音频格式**：WAV, MP3, FLAC等（通过soundfile库支持）

### 1. 获取系统状态

**接口地址**：`GET /status`

**请求示例**：
```bash
curl -X GET "http://localhost:8051/status"
```

**响应示例**：
```json
{
  "status": "running",
  "character": "dw",
  "audio_model": "checkpoint/audio.pkl",
  "render_model": "checkpoint/15000.pth",
  "virtual_camera": true,
  "queue_size": 0
}
```

**响应字段说明**：
- `status`：系统运行状态（running/stopped）
- `character`：当前使用的角色
- `audio_model`：音频模型路径
- `render_model`：渲染模型路径
- `virtual_camera`：虚拟摄像头是否可用
- `queue_size`：当前音频处理队列大小

### 2. 接收音频文件

**接口地址**：`POST /receive_audio`

**请求方式**：multipart/form-data

**请求参数**：
- `audio_file`：音频文件（必需）

**请求示例**：
```bash
curl -X POST "http://localhost:8051/receive_audio" \
  -F "audio_file=@test_audio.wav"
```

**Python示例**：
```python
import requests

url = "http://localhost:8051/receive_audio"
with open("test_audio.wav", "rb") as f:
    files = {"audio_file": ("test_audio.wav", f, "audio/wav")}
    response = requests.post(url, files=files)
    print(response.json())
```

**响应示例**：
```json
{
  "status": "success",
  "message": "Audio received and queued for processing",
  "audio_info": {
    "sample_rate": 16000,
    "duration": 3.5,
    "samples": 56000
  }
}
```

### 3. 接收Base64编码音频

**接口地址**：`POST /receive_audio_base64`

**请求方式**：application/json

**请求参数**：
```json
{
  "audio_data": "base64编码的音频数据",
  "sample_rate": 16000
}
```

**请求示例**：
```bash
curl -X POST "http://localhost:8051/receive_audio_base64" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
    "sample_rate": 16000
  }'
```

**Python示例**：
```python
import requests
import base64
import soundfile as sf

# 读取音频文件并转换为base64
audio_data, sample_rate = sf.read("test_audio.wav")
audio_bytes = sf.write(None, audio_data, sample_rate, format='WAV')
audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

url = "http://localhost:8051/receive_audio_base64"
data = {
    "audio_data": audio_base64,
    "sample_rate": sample_rate
}
response = requests.post(url, json=data)
print(response.json())
```

**响应示例**：
```json
{
  "status": "success",
  "message": "Base64 audio received and queued for processing",
  "audio_info": {
    "sample_rate": 16000,
    "duration": 3.5,
    "samples": 56000
  }
}
```

## 错误处理

### 常见错误码

- **400 Bad Request**：请求参数错误
- **422 Unprocessable Entity**：音频文件格式不支持或数据无效
- **500 Internal Server Error**：服务器内部错误

### 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

### 常见问题及解决方案

1. **音频文件上传失败**
   - 检查文件格式是否支持
   - 确认文件大小不超过限制
   - 验证文件是否损坏

2. **Base64音频解码失败**
   - 检查Base64编码是否正确
   - 确认音频数据格式
   - 验证采样率参数

3. **系统无响应**
   - 检查DH-LIVE服务是否正常运行
   - 验证网络连接
   - 查看服务器日志

## 集成示例

### AI-VTUBER集成示例

```python
import requests
import base64
from io import BytesIO
import soundfile as sf

class DHLiveClient:
    def __init__(self, base_url="http://localhost:8051"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_status(self):
        """检查DH-LIVE系统状态"""
        try:
            response = self.session.get(f"{self.base_url}/status")
            return response.json()
        except Exception as e:
            print(f"状态检查失败: {e}")
            return None
    
    def send_audio_file(self, audio_path):
        """发送音频文件"""
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio_file': (audio_path, f, 'audio/wav')}
                response = self.session.post(
                    f"{self.base_url}/receive_audio",
                    files=files
                )
                return response.json()
        except Exception as e:
            print(f"音频文件发送失败: {e}")
            return None
    
    def send_audio_base64(self, audio_data, sample_rate=16000):
        """发送Base64编码音频"""
        try:
            # 将音频数据转换为WAV格式的字节流
            buffer = BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            audio_bytes = buffer.getvalue()
            
            # 编码为Base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            data = {
                "audio_data": audio_base64,
                "sample_rate": sample_rate
            }
            
            response = self.session.post(
                f"{self.base_url}/receive_audio_base64",
                json=data
            )
            return response.json()
        except Exception as e:
            print(f"Base64音频发送失败: {e}")
            return None

# 使用示例
client = DHLiveClient()

# 检查系统状态
status = client.check_status()
if status and status.get('status') == 'running':
    print("DH-LIVE系统运行正常")
    
    # 发送音频文件
    result = client.send_audio_file("test_audio.wav")
    if result and result.get('status') == 'success':
        print("音频发送成功，开始渲染")
else:
    print("DH-LIVE系统未运行")
```

## 性能优化建议

### 1. 音频格式优化

- **推荐采样率**：16000Hz（系统默认）
- **推荐格式**：WAV（无损，处理速度快）
- **音频长度**：建议单次发送不超过10秒

### 2. 网络优化

- 使用HTTP连接池（如示例中的`requests.Session()`）
- 对于频繁调用，考虑使用异步请求
- 实现重试机制处理网络异常

### 3. 系统资源

- 确保GPU内存充足（推荐8GB以上）
- 监控CPU使用率，避免过载
- 定期检查音频处理队列大小

## 命令行参数

启动DH-LIVE时可以使用以下参数：

```bash
python dh_live_realtime.py [OPTIONS]
```

**可用参数**：

- `--audio_model`：音频模型路径（默认：checkpoint/audio.pkl）
- `--render_model`：渲染模型路径（默认：checkpoint/15000.pth）
- `--character`：角色名称（默认：dw）
- `--host`：服务主机地址（默认：localhost）
- `--port`：服务端口（默认：8051）

**示例**：
```bash
# 使用自定义角色和端口
python dh_live_realtime.py --character alice --port 8052

# 使用自定义模型路径
python dh_live_realtime.py --audio_model models/audio_v2.pkl --render_model models/render_v2.pth
```

## 故障排除

### 1. 虚拟摄像头问题

如果虚拟摄像头无法正常工作：

```bash
# 安装虚拟摄像头驱动
虚拟摄像头驱动.bat

# 检查pyvirtualcam安装
pip install pyvirtualcam
```

### 2. GPU相关问题

```bash
# 检查CUDA环境
查看cuda版本.bat

# 运行GPU诊断
GPU诊断.bat
```

### 3. 依赖问题

```bash
# 安装所有依赖
pip install -r requirements.txt

# 检查环境
python check_env.py
```

## 技术支持

如遇到问题，请检查：

1. **系统日志**：查看控制台输出的详细信息
2. **模型文件**：确认checkpoint目录下的模型文件完整
3. **环境配置**：验证Python环境和依赖库版本
4. **硬件要求**：确保GPU内存和计算能力满足要求

---

**更新日期**：2025年1月
**版本**：v2.0
**兼容性**：DH-LIVE v2.0+