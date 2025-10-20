#!/usr/bin/env python3
"""
内存监控脚本 - 用于监控 DH-LIVE 系统的资源使用情况
"""

import psutil
import time
import matplotlib.pyplot as plt
from collections import deque
import threading

class MemoryMonitor:
    def __init__(self, process_name="python", max_points=300):
        self.process_name = process_name
        self.max_points = max_points
        self.memory_data = deque(maxlen=max_points)
        self.cpu_data = deque(maxlen=max_points)
        self.time_data = deque(maxlen=max_points)
        self.is_monitoring = False
        self.target_pid = None
        
    def find_dh_live_process(self):
        """查找 DH-LIVE 进程"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'dh_live_realtime.py' in ' '.join(proc.info['cmdline'] or []):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def start_monitoring(self):
        """开始监控"""
        self.is_monitoring = True
        self.target_pid = self.find_dh_live_process()
        
        if not self.target_pid:
            print("未找到 DH-LIVE 进程，监控所有 Python 进程...")
        else:
            print(f"找到 DH-LIVE 进程 PID: {self.target_pid}")
        
        def monitor_loop():
            start_time = time.time()
            while self.is_monitoring:
                try:
                    if self.target_pid:
                        # 监控特定进程
                        proc = psutil.Process(self.target_pid)
                        memory_mb = proc.memory_info().rss / 1024 / 1024
                        cpu_percent = proc.cpu_percent()
                    else:
                        # 监控所有 Python 进程的总和
                        total_memory = 0
                        total_cpu = 0
                        python_procs = []
                        
                        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                            try:
                                if 'python' in proc.info['name'].lower():
                                    python_procs.append(proc)
                                    total_memory += proc.info['memory_info'].rss
                                    total_cpu += proc.info['cpu_percent'] or 0
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                        
                        memory_mb = total_memory / 1024 / 1024
                        cpu_percent = total_cpu
                    
                    current_time = time.time() - start_time
                    
                    self.memory_data.append(memory_mb)
                    self.cpu_data.append(cpu_percent)
                    self.time_data.append(current_time)
                    
                    # 打印当前状态
                    print(f"时间: {current_time:.1f}s, 内存: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
                    
                    # 检查内存泄漏警告
                    if memory_mb > 4000:  # 超过4GB
                        print(f"⚠️ 内存使用过高: {memory_mb:.1f}MB")
                    
                    time.sleep(1)
                    
                except psutil.NoSuchProcess:
                    print("目标进程已退出")
                    break
                except Exception as e:
                    print(f"监控错误: {e}")
                    time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
    
    def plot_results(self):
        """绘制监控结果"""
        if not self.memory_data:
            print("没有监控数据可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 内存使用图
        ax1.plot(list(self.time_data), list(self.memory_data), 'b-', linewidth=2)
        ax1.set_ylabel('内存使用 (MB)')
        ax1.set_title('DH-LIVE 系统资源监控')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=4000, color='r', linestyle='--', alpha=0.7, label='4GB 警告线')
        ax1.legend()
        
        # CPU使用图
        ax2.plot(list(self.time_data), list(self.cpu_data), 'g-', linewidth=2)
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('CPU 使用率 (%)')
        ax2.grid(True, alpha=0.3)
        
        # 统计信息
        max_memory = max(self.memory_data)
        avg_memory = sum(self.memory_data) / len(self.memory_data)
        avg_cpu = sum(self.cpu_data) / len(self.cpu_data)
        
        plt.figtext(0.02, 0.02, 
                   f"统计信息 - 最大内存: {max_memory:.1f}MB, 平均内存: {avg_memory:.1f}MB, 平均CPU: {avg_cpu:.1f}%",
                   fontsize=10)
        
        plt.tight_layout()
        plt.savefig('dh_live_monitor.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n=== 监控报告 ===")
        print(f"最大内存使用: {max_memory:.1f}MB")
        print(f"平均内存使用: {avg_memory:.1f}MB")
        print(f"平均CPU使用: {avg_cpu:.1f}%")
        print(f"监控时长: {max(self.time_data):.1f}秒")

def main():
    monitor = MemoryMonitor()
    print("开始监控 DH-LIVE 系统资源使用...")
    print("按 Ctrl+C 停止监控并查看报告")
    
    try:
        monitor.start_monitoring()
        
        # 保持主线程运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止监控...")
        monitor.stop_monitoring()
        monitor.plot_results()

if __name__ == "__main__":
    main()