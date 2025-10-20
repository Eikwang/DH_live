#!/usr/bin/env python3
"""
简单的内存监控脚本
"""

import psutil
import time
import requests

def monitor_dh_live_memory():
    """监控DH-LIVE内存使用"""
    print("=== DH-LIVE 内存监控 ===")
    print("每10秒检查一次内存使用情况")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            # 获取系统状态
            try:
                response = requests.get("http://localhost:8000/status", timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    memory_mb = status.get('performance', {}).get('memory_mb', 0)
                    fps = status.get('performance', {}).get('fps', 0)
                    queue_size = status.get('audio_queue_size', 0)
                    
                    current_time = time.strftime("%H:%M:%S")
                    
                    # 根据内存使用情况显示不同颜色的警告
                    if memory_mb > 1800:
                        status_icon = "🔴"
                        status_text = "严重警告"
                    elif memory_mb > 1500:
                        status_icon = "🟡"
                        status_text = "警告"
                    elif memory_mb > 1200:
                        status_icon = "🟠"
                        status_text = "注意"
                    else:
                        status_icon = "🟢"
                        status_text = "正常"
                    
                    print(f"[{current_time}] {status_icon} {status_text} - 内存: {memory_mb:.1f}MB, FPS: {fps:.1f}, 队列: {queue_size}")
                    
                    # 如果内存过高，提供清理建议
                    if memory_mb > 1800:
                        print("   建议执行强制清理: curl -X POST http://localhost:8000/cleanup_memory")
                    
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] ❌ 无法获取系统状态")
                    
            except requests.exceptions.ConnectionError:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ 连接失败 - DH-LIVE可能未运行")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ 错误: {e}")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n监控停止")

if __name__ == "__main__":
    monitor_dh_live_memory()