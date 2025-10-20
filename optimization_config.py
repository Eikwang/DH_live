# DH-LIVE 系统优化配置
# 本文件包含优化后的系统配置参数

import json

# 优化配置
OPTIMIZED_CONFIG = {
    "memory_management": {
        "max_queue_size": 3,           # 音频队列大小（从10减少到3）
        "max_memory_mb": 4000,         # 最大内存使用限制(MB)
        "gc_interval": 30,             # 垃圾回收间隔(秒)
        "memory_check_interval": 5,    # 内存检查间隔(秒)
        "idle_static": True,           # 启用静态帧复用
        "frame_pool_size": 5           # 帧池大小
    },
    
    "performance": {
        "target_fps": 25,              # 目标帧率
        "processing_times_maxlen": 5,  # 处理时间历史记录数量（从10减少到5）
        "memory_history_maxlen": 3,    # 内存使用历史记录数量（从2增加到3）
        "audio_timeout": 0.01,         # 音频队列超时时间(秒，从0.02减少到0.01)
        "silence_timeout": 0.1,        # 静默超时时间(秒)
        "transition_frames": 5         # 过渡帧数
    },
    
    "debug": {
        "debug_mode": False,           # 关闭调试模式以减少日志输出
        "metrics_print_interval": 20, # 性能统计打印间隔(秒，从10增加到20)
        "log_level": "INFO"           # 日志级别
    },
    
    "optimization_features": {
        "frame_pooling": True,         # 启用帧池
        "memory_monitoring": True,     # 启用内存监控
        "aggressive_cleanup": True,    # 启用激进清理
        "cache_management": True,      # 启用缓存管理
        "weak_references": True        # 使用弱引用
    }
}

def save_config(filename="optimized_config.json"):
    """保存优化配置到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(OPTIMIZED_CONFIG, f, indent=2, ensure_ascii=False)
    print(f"配置已保存到: {filename}")

def load_config(filename="optimized_config.json"):
    """从文件加载优化配置"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"配置已从 {filename} 加载")
        return config
    except FileNotFoundError:
        print(f"配置文件 {filename} 不存在，使用默认配置")
        return OPTIMIZED_CONFIG

if __name__ == "__main__":
    # 保存默认优化配置
    save_config()
    
    print("=== DH-LIVE 系统优化配置 ===")
    print(json.dumps(OPTIMIZED_CONFIG, indent=2, ensure_ascii=False))