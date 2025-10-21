import time
import os
import numpy as np
import cv2
import threading
import queue
import json
from io import BytesIO
import soundfile as sf
import pyvirtualcam
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import subprocess
import argparse
import base64
import psutil
import gc
import torch
from collections import deque
import weakref
from threading import Lock

class DHLiveRealtime:
    def __init__(self, audio_model_path="checkpoint/audio.pkl", render_model_path="checkpoint/15000.pth", character="dw", auto_start=True):
        # 初始化音频和渲染模型
        self.audioModel = AudioModel()
        self.audioModel.loadModel(audio_model_path)
        
        self.renderModel = RenderModel()
        self.renderModel.loadModel(render_model_path)
        
        # 设置角色
        self.character = character
        pkl_path = "video_data/{}/keypoint_rotate.pkl".format(character)
        self.video_path = "video_data/{}/circle.mp4".format(character)
        self.renderModel.reset_charactor(self.video_path, pkl_path)
        
        # 动态获取角色视频的实际尺寸
        self.video_width, self.video_height = self.get_video_dimensions(self.video_path)
        
        # 新增：空闲时播放的视频（与circle.mp4同目录）
        self.idle_video_path = os.path.join(os.path.dirname(self.video_path), "silent.mp4")
        
        # 提前设置debug属性
        self.debug = False       # 调试日志开关
        
        self.idle_video_cap = None
        self.idle_video_fps = None
        self._idle_video_enabled = False
        # 初始化空闲视频（若存在）
        self._init_idle_video()
        
        # 音频参数
        self.sample_rate = 16000
        
        # 音频队列和控制 - 大幅减少队列大小防止内存泄漏
        self.max_queue_size = 3  # 减少到3个，防止音频堆积
        self.audio_queue = queue.Queue(maxsize=self.max_queue_size)
        self.is_running = False
        self.virtual_cam = None
        
        # 线程安全锁
        self.frame_lock = Lock()
        self.state_lock = Lock()
        
        # 性能监控 - 减少内存占用
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.processing_times = deque(maxlen=5)  # 减少到5次
        self.memory_usage_history = deque(maxlen=3)  # 减少到3次历史记录
        
        # 帧缓存池 - 重用帧对象
        self.frame_pool = deque(maxlen=5)
        self.frame_pool_lock = Lock()
        
        # 帧处理缓存
        self._last_frame_size = None
        self._resize_params = None
        
        # 性能/行为配置
        self.target_fps = 25
        # 禁用静态帧复用：无音频输入直接播放 idle 视频
        self.idle_static = False   # 无音频时直接使用 idle 视频
        self.render_output_color = "BGR"  # 渲染模型输出的颜色空间
        
        # 自适应平滑配置（音频能量驱动）
        self.ema_prev_mouth = None
        self.ema_alpha_base = 0.7
        self.ema_alpha_low = 0.55
        self.ema_alpha_high = 0.85
        self.ema_energy_low = 0.01
        self.ema_energy_high = 0.05
        
        # 根据target_fps计算每帧采样数（需在default_silent_chunk之前）
        self.samples_per_read = int(round(self.sample_rate / self.target_fps))
        
        # 优化的缓存管理
        self.default_silent_chunk = np.zeros(self.samples_per_read, dtype=np.float32)
        self._cached_idle_mouth = None
        self._cached_idle_frame = None
        self._cached_frame_refs = weakref.WeakSet()  # 使用弱引用避免循环引用
        self._last_metrics_print = time.time()
        
        # 内存管理配置 - 更严格的限制
        self.gc_interval = 180  # 垃圾回收间隔（秒）
        self.memory_check_interval = 60  # 内存检查间隔（秒）
        self.max_memory_mb = 1500  # 最大内存使用限制（MB）- 降低到1.5GB
        self.last_memory_check = time.time()
        
        # 更激进的内存清理配置
        self.force_cleanup_threshold = 1500  # 1.5GB时强制清理（原2GB）
        self.regular_cleanup_threshold = 1400  # 1.2GB时常规清理（原1.5GB）
        self.light_cleanup_threshold = 1300   # 1GB时轻度清理（原1.2GB）
        
        # 播放状态管理与平滑过渡
        self.silence_timeout = 0.1  # s，超过该静默时间切换为idle状态
        self.current_state = "idle"  # idle | audio
        self._last_audio_event_time = 0.0
        self.was_silent = False
        # 适度增加过渡帧数以更柔和（由6改为10）
        self.transition_duration_frames = 5  # 过渡帧数
        self._transition_remaining = 0
        self._transition_from = None  # 过渡起始帧（BGR，已按目标尺寸）
        self._last_output_frame = None  # 最近一次输出的帧（BGR，已按目标尺寸）
        # 引入统一时间轴，用于idle视频相位对齐
        self.timeline_start_time = time.time()
        
        self.idle_video_cap = None
        self.idle_video_fps = None
        self.idle_video_frame_count = 0
        self._idle_video_enabled = False
        
        # 预分配的帧缓冲区
        self._preallocated_frames = {}
        
        # 初始化空闲视频
        self._init_idle_video()
        
        # 如果设置了自动启动，则启动系统
        if auto_start:
            success, message = self.start_system()
            if success:
                print(f"系统自动启动成功: {message}")
            else:
                print(f"系统自动启动失败: {message}")

    def get_frame_from_pool(self, shape):
        """从帧池中获取或创建帧对象"""
        with self.frame_pool_lock:
            if self.frame_pool:
                frame = self.frame_pool.popleft()
                if frame.shape == shape:
                    frame.fill(0)  # 清零复用
                    return frame
                # 尺寸不匹配，继续创建新帧
            return np.zeros(shape, dtype=np.uint8)
    
    def return_frame_to_pool(self, frame):
        """将帧返回到池中供复用"""
        if frame is None:
            return
        with self.frame_pool_lock:
            if len(self.frame_pool) < self.frame_pool.maxlen:
                self.frame_pool.append(frame)

    def _preallocate_frame_buffer(self, key, shape):
        """预分配特定尺寸的帧缓冲区"""
        if key not in self._preallocated_frames:
            self._preallocated_frames[key] = np.zeros(shape, dtype=np.uint8)
        return self._preallocated_frames[key]

    def _init_idle_video(self):
        """初始化空闲视频播放"""
        try:
            if os.path.exists(self.idle_video_path):
                self.idle_video_cap = cv2.VideoCapture(self.idle_video_path)
                if self.idle_video_cap.isOpened():
                    self.idle_video_fps = self.idle_video_cap.get(cv2.CAP_PROP_FPS)
                    self.idle_video_frame_count = int(self.idle_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._idle_video_enabled = True
                    if not self.debug:  # 只在非调试模式下显示重要信息
                        print(f"Idle video enabled: {self.idle_video_path} @ {self.idle_video_fps:.2f} fps, frames={self.idle_video_frame_count}")
                else:
                    if not self.debug:
                        print(f"Failed to open idle video: {self.idle_video_path}")
            else:
                if not self.debug:
                    print(f"Idle video not found: {self.idle_video_path}")
        except Exception as e:
            if not self.debug:
                print(f"Idle video init error: {e}")

    def _cleanup_cached_frames(self):
        """清理缓存帧以释放内存 - 加强版本"""
        try:
            # 清理静态缓存
            if not self.idle_static:
                self._cached_idle_frame = None
                self._cached_idle_mouth = None
            
            # 清理预分配缓冲区（更激进）
            if len(self._preallocated_frames) > 2:  # 减少到2个
                # 保留最近使用的2个尺寸的缓冲区
                keys_to_remove = list(self._preallocated_frames.keys())[2:]
                for key in keys_to_remove:
                    del self._preallocated_frames[key]
            
            # 清理弱引用集合
            self._cached_frame_refs.clear()
            
            # 清理帧池
            with self.frame_pool_lock:
                self.frame_pool.clear()
            
            # 清理处理时间历史
            self.processing_times.clear()
            self.memory_usage_history.clear()
            
        except Exception as e:
            if self.debug:
                print(f"Frame cleanup error: {e}")

    def _force_memory_cleanup(self):
        """强制内存清理 - 根本性优化版本"""
        try:
            print("执行深度内存清理...")
            
            # 1. 清理所有缓存
            self._cleanup_cached_frames()
            
            # 2. 强制清理静态缓存（无论配置）
            self._cached_idle_frame = None
            self._cached_idle_mouth = None
            self._last_output_frame = None
            self._transition_from = None
            
            # 3. 清空所有预分配缓冲区
            self._preallocated_frames.clear()
            
            # 4. 重置缓存参数
            self._last_frame_size = None
            self._resize_params = None
            
            # 5. 清理音频队列中的所有数据
            cleared_count = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    cleared_count += 1
                except queue.Empty:
                    break
            if cleared_count > 0:
                print(f"从音频队列清理了 {cleared_count} 个项目")
            
            # 6. 清理帧池
            with self.frame_pool_lock:
                pool_size = len(self.frame_pool)
                self.frame_pool.clear()
                if pool_size > 0:
                    print(f"清理了 {pool_size} 个帧池对象")
            
            # 7. 清理历史数据
            self.processing_times.clear()
            self.memory_usage_history.clear()
            
            # 8. 重新初始化缓存对象
            self.default_silent_chunk = np.zeros(self.samples_per_read, dtype=np.float32)
            
            # 9. 深度GPU内存清理（关键！）
            try:
                import torch
                if torch.cuda.is_available():
                    # 强制清理GPU缓存
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # 清理进程间共享内存
                    torch.cuda.ipc_collect()
                    
                    # 重置所有GPU内存统计
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                    
                    # 强制重置AI模型的内部状态
                    if hasattr(self.audioModel, 'reset'):
                        self.audioModel.reset()
                        print("AudioModel 状态已重置")
                    
                    # 清理PyTorch JIT缓存
                    if hasattr(torch.jit, '_clear_class_registry'):
                        torch.jit._clear_class_registry()
                    
                    print("GPU内存深度清理完成")
            except ImportError:
                pass
            
            # 10. 多次垃圾回收并等待系统稳定
            for i in range(7):  # 增加到7次
                gc.collect()
                time.sleep(0.05)  # 给系统更多时间稳定
            
            print("深度内存清理完成")
            
        except Exception as e:
            print(f"深度清理错误: {e}")

    def _read_idle_video_frame(self):
        try:
            if not self._idle_video_enabled or self.idle_video_cap is None:
                return None
            ret, frame = self.idle_video_cap.read()
            if not ret:
                # 到达末尾后循环播放
                self.idle_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.idle_video_cap.read()
                if not ret:
                    return None
            return frame
        except Exception as e:
            if self.debug:
                print(f"Idle video read error: {e}")
            return None

    def get_video_dimensions(self, video_path):
        """返回给定视频的(width, height)，默认为256x256。"""
        try:
            if not os.path.exists(video_path):
                # 初始化时debug可能还没设置，使用getattr获取默认值
                if getattr(self, 'debug', False):
                    print(f"Video not found for dimension probe: {video_path}")
                return 256, 256
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if getattr(self, 'debug', False):
                    print(f"Failed to open video for dimension probe: {video_path}")
                return 256, 256
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w <= 0 or h <= 0:
                return 256, 256
            return w, h
        except Exception as e:
            if getattr(self, 'debug', False):
                print(f"get_video_dimensions error: {e}")
            return 256, 256

    def init_virtual_camera(self):
        """初始化虚拟摄像头，尝试多种后端"""
        print(f"正在初始化虚拟摄像头: {self.video_width}x{self.video_height}@{self.target_fps}fps")
        
        # 尝试不同的虚拟摄像头后端
        backends_to_try = []
        
        # 检查可用的后端
        try:
            import platform
            if platform.system() == "Windows":
                # Windows系统优先尝试OBS Virtual Camera
                backends_to_try = ['obs', 'unitycapture', None]  # None表示默认后端
            else:
                backends_to_try = [None]  # 其他系统使用默认后端
        except:
            backends_to_try = [None]
        
        for backend in backends_to_try:
            try:
                print(f"尝试使用后端: {backend if backend else 'default'}")
                
                if backend:
                    self.virtual_cam = pyvirtualcam.Camera(
                        width=self.video_width,
                        height=self.video_height,
                        fps=self.target_fps,
                        fmt=pyvirtualcam.PixelFormat.BGR,
                        backend=backend
                    )
                else:
                    self.virtual_cam = pyvirtualcam.Camera(
                        width=self.video_width,
                        height=self.video_height,
                        fps=self.target_fps,
                        fmt=pyvirtualcam.PixelFormat.BGR,
                    )
                
                print(f"虚拟摄像头初始化成功 (backend: {backend if backend else 'default'}): {self.video_width}x{self.video_height}@{self.target_fps}fps, fmt=BGR")

                # 发送一帧绿色测试图（BGR），并对齐帧率
                test_frame = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
                test_frame[:] = [0, 255, 0]
                cv2.putText(
                    test_frame,
                    "DH-LIVE Ready",
                    (max(20, self.video_width // 10), max(40, self.video_height // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                
                # 尝试发送测试帧
                try:
                    self.virtual_cam.send(np.ascontiguousarray(test_frame))
                    self.virtual_cam.sleep_until_next_frame()
                    print("测试帧发送成功")
                    return True
                except Exception as send_error:
                    print(f"测试帧发送失败: {send_error}")
                    # 继续尝试下一个后端
                    if self.virtual_cam:
                        try:
                            self.virtual_cam.close()
                        except:
                            pass
                        self.virtual_cam = None
                    continue
                    
            except Exception as e:
                print(f"后端 {backend if backend else 'default'} 初始化失败: {e}")
                if self.virtual_cam:
                    try:
                        self.virtual_cam.close()
                    except:
                        pass
                    self.virtual_cam = None
                continue
        
        print("所有虚拟摄像头后端都初始化失败")
        print("请确保:")
        print("1. 已安装OBS Studio并启用Virtual Camera插件")
        print("2. 或者安装了Unity Capture等虚拟摄像头软件")
        print("3. 没有其他程序正在使用虚拟摄像头")
        return False

    def resize_and_pad(self, frame, target_width, target_height):
        """优化的帧缩放和填充方法，减少内存分配"""
        h, w = frame.shape[:2]
        
        # 如果尺寸已经匹配，则直接返回
        if h == target_height and w == target_width:
            return frame

        # 检查是否可以使用缓存的参数
        current_frame_size = (h, w)
        cache_key = f"{target_width}x{target_height}"
        
        if (self._last_frame_size != current_frame_size or 
            self._resize_params is None):
            
            frame_aspect = w / h
            target_aspect = target_width / target_height

            # 根据宽高比计算新的尺寸
            if frame_aspect > target_aspect:
                # 帧比目标宽，根据宽度缩放
                new_w = target_width
                new_h = int(new_w / frame_aspect)
            else:
                # 帧比目标高，根据高度缩放
                new_h = target_height
                new_w = int(new_h * frame_aspect)

            # 计算偏移量，使其居中
            y_offset = (target_height - new_h) // 2
            x_offset = (target_width - new_w) // 2
            
            # 缓存参数
            self._last_frame_size = current_frame_size
            self._resize_params = (new_w, new_h, x_offset, y_offset)
        else:
            # 使用缓存的参数
            new_w, new_h, x_offset, y_offset = self._resize_params

        # 缩放图像
        resized = cv2.resize(frame, (new_w, new_h))

        # 防止预分配缓冲区过度增长
        if len(self._preallocated_frames) > 2:
            # 只保疙2个尺寸，其他直接创建
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        else:
            # 使用预分配的缓冲区或从池中获取
            padded = self._preallocate_frame_buffer(cache_key, (target_height, target_width, 3))
            padded.fill(0)  # 清零黑色背景
        
        # 将缩放后的图像粘贴到画布中央
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # 清理中间变量
        del resized
        
        return padded

    def _resample_audio(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """高保真重采样（优先使用resample_poly，回退到np.interp）。"""
        try:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(src_rate, dst_rate)
            up = dst_rate // g
            down = src_rate // g
            audio_f32 = audio.astype(np.float32, copy=False)
            res = resample_poly(audio_f32, up, down)
            return res.astype(np.float32, copy=False)
        except Exception:
            # 回退到线性插值，确保长度匹配且保持浮点精度
            ratio = float(dst_rate) / float(src_rate)
            new_length = int(round(len(audio) * ratio))
            x = np.arange(len(audio), dtype=np.float32)
            xp = np.linspace(0, len(audio) - 1, new_length, dtype=np.float32)
            audio_f32 = audio.astype(np.float32, copy=False)
            res = np.interp(xp, x, audio_f32)
            return res.astype(np.float32, copy=False)

    def process_audio_chunk(self, audio_chunk):
        """处理音频块，生成嘴型帧 - 根本性内存优化版本"""
        try:
            # 确保音频数据格式正确
            if len(audio_chunk) != self.samples_per_read:
                # 如果音频块长度不匹配，进行重采样或填充
                if len(audio_chunk) > self.samples_per_read:
                    audio_chunk = audio_chunk[:self.samples_per_read]
                else:
                    # 填充零值
                    padding = np.zeros(self.samples_per_read - len(audio_chunk))
                    audio_chunk = np.concatenate([audio_chunk, padding])
            

            
            # 使用torch.no_grad()确保不产生梯度，避免内存泄漏
            with torch.no_grad():
                # 生成嘴型帧
                mouth_frame = self.audioModel.interface_frame(audio_chunk.astype(np.float32))
            
            # 基于块能量的自适应EMA平滑，兼顾响应与顺滑
            try:
                energy = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))
                alpha = self.ema_alpha_base
                if energy < self.ema_energy_low:
                    alpha = self.ema_alpha_low
                elif energy > self.ema_energy_high:
                    alpha = self.ema_alpha_high
                if isinstance(mouth_frame, np.ndarray):
                    if self.ema_prev_mouth is None or getattr(self.ema_prev_mouth, 'shape', None) != getattr(mouth_frame, 'shape', None):
                        self.ema_prev_mouth = mouth_frame.copy()
                    else:
                        mouth_frame = (alpha * mouth_frame + (1.0 - alpha) * self.ema_prev_mouth).astype(mouth_frame.dtype, copy=False)
                        self.ema_prev_mouth = mouth_frame.copy()
            except Exception:
                # 出现异常则跳过平滑，保持原始输出
                pass
            
            # 立即清理输入数据引用
            del audio_chunk
            

            
            return mouth_frame
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None
    
    def render_frame(self, mouth_frame):
        try:
            import torch
            with torch.no_grad():
                frame = self.renderModel.interface(mouth_frame)
            del mouth_frame
            return frame
        except Exception as e:
            print(f"Rendering error: {e}")
            return None
    
    def render_default_frame(self):
        """渲染默认静态帧（无音频输入时） - 优化版本"""
        try:
            # 启用静态缓存复用，大幅减少CPU消耗
            if self.idle_static and self._cached_idle_frame is not None:
                return self._cached_idle_frame.copy()  # 返回副本防止修改
            
            if self._cached_idle_mouth is None:
                # 仅生成一次静音嘴型
                self._cached_idle_mouth = self.audioModel.interface_frame(self.default_silent_chunk)
            
            frame = self.renderModel.interface(self._cached_idle_mouth)
            
            if self.idle_static and frame is not None:
                # 缓存静态帧（副本）
                self._cached_idle_frame = frame.copy()
                
            return frame
        except Exception as e:
            if self.debug:
                print(f"Default frame rendering error: {e}")
            return None

    def _sync_idle_video_to_now(self):
        """在切换到idle时，将idle视频的播放位置与当前时间轴对齐，减轻突兀感"""
        try:
            if not self._idle_video_enabled or self.idle_video_cap is None or self.idle_video_frame_count <= 0:
                return
            elapsed = time.time() - self.timeline_start_time
            # 按目标FPS计算期望帧索引，并对齐至视频总帧数
            idx = int(round(elapsed * self.target_fps)) % self.idle_video_frame_count
            # 定位到对应帧（注意某些解码器可能只保证关键帧附近准确）
            self.idle_video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        except Exception as e:
            if self.debug:
                print(f"Idle video sync error: {e}")

    def _maybe_switch_state(self, target_state: str):
        """根据目标状态决定是否切换，并准备过渡参数"""
        if target_state not in ("idle", "audio"):
            return
        if target_state != self.current_state:
            if self.debug:
                print(f"State change: {self.current_state} -> {target_state}")
            self.current_state = target_state
            # 动态过渡：静默恢复到音频时使用更短的过渡以降低口型滞后
            if target_state == "audio" and getattr(self, 'was_silent', False):
                self._transition_remaining = min(3, self.transition_duration_frames)
            else:
                self._transition_remaining = self.transition_duration_frames
            # 将过渡起点设为上一帧输出（若存在）
            self._transition_from = self._last_output_frame
            # 在切换到idle时做相位同步，使silent.mp4从与当前时间相匹配的帧开始
            if target_state == "idle":
                self._sync_idle_video_to_now()

    def send_frame_to_camera(self, frame):
        """优化的帧发送方法，深度减少内存泄漏"""
        try:
            if self.virtual_cam is None or frame is None:
                return
            
            # 确保帧的尺寸和宽高比正确
            resized_frame = self.resize_and_pad(frame, self.video_width, self.video_height)
            
            # 立即清理原始帧引用（避免numpy数组比较歧义）
            if frame is not None and not np.array_equal(frame, resized_frame):
                del frame
            
            # 确保类型与内存布局满足要求（uint8，连续内存）
            if resized_frame.dtype != np.uint8:
                resized_frame = resized_frame.astype(np.uint8)
            
            # 使用现有内存布局或创建连续内存
            if not resized_frame.flags['C_CONTIGUOUS']:
                resized_frame = np.ascontiguousarray(resized_frame)
            
            # 发送到虚拟摄像头（BGR 像素格式）
            self.virtual_cam.send(resized_frame)
            self.virtual_cam.sleep_until_next_frame()
            
            # 立即清理处理后的帧
            del resized_frame
            
            # 更新帧计数
            self.frame_count += 1
            
        except Exception as e:
            print(f"Frame sending error: {e}")

    def send_with_transition(self, new_frame):
        """发送帧并在状态切换时进行短暂的渐变过渡 - 深度优化"""
        if new_frame is None or self.virtual_cam is None:
            return
            
        # 先调整到目标尺寸（BGR）
        prepared_new = self.resize_and_pad(new_frame, self.video_width, self.video_height)
        
        # 立即清理原始输入帧（避免numpy数组比较歧义）
        if new_frame is not None and not np.array_equal(new_frame, prepared_new):
            del new_frame
        
        frame_to_send = prepared_new
        
        if self._transition_remaining > 0 and self._transition_from is not None:
            try:
                 prev = self._transition_from
                 # 确保上一帧与新帧尺寸一致
                 if prev is None or prev.shape[:2] != prepared_new.shape[:2]:
                     prev = self.resize_and_pad(prev if prev is not None else prepared_new, self.video_width, self.video_height)
                 
                 # 计算当前过渡进度
                 step_idx = self.transition_duration_frames - self._transition_remaining + 1
                 linear = max(0.0, min(1.0, step_idx / self.transition_duration_frames))
                 # 使用smoothstep缓入缓出，观感更自然
                 progress = linear * linear * (3 - 2 * linear)
                 
                 # 创建混合帧
                 blended_frame = cv2.addWeighted(prev.astype(np.float32), 1.0 - progress,
                                                prepared_new.astype(np.float32), progress, 0.0).astype(np.uint8)
                 frame_to_send = blended_frame
                 
                 self._transition_remaining -= 1
                 if self._transition_remaining == 0:
                     self._transition_from = None
                     
            except Exception as e:
                if self.debug:
                    print(f"Transition blend error: {e}")
                # 过渡失败则直接发送新帧
                frame_to_send = prepared_new
        
        # 真正发送
        self.send_frame_to_camera(frame_to_send)
        
        # 记录最近一次输出帧（用于后续过渡/持帧）- 使用浅拷贝
        if prepared_new is not None:
            self._last_output_frame = prepared_new.copy()
        
        # 清理中间变量（避免numpy数组比较歧义）
        if frame_to_send is not None and prepared_new is not None and not np.array_equal(frame_to_send, prepared_new):
            del frame_to_send
        del prepared_new
    
    def update_performance_metrics(self):
        """更新性能监控指标 - 优化版本"""
        try:
            current_time = time.time()
            
            # 计算FPS，并将较重的统计放到每秒执行一次的分支里
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
                
                # 优化：降低内存采样频率
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.memory_usage_history.append(memory_mb)
                except:
                    pass  # 忽略内存采集错误
                
                # 优化：每20秒打印一次性能统计（减少日志输出）
                if current_time - self._last_metrics_print >= 20.0:  # 增加到20秒
                    avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                    max_processing_time = max(self.processing_times) if self.processing_times else 0
                    avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0
                    
                    # 优化：只在非调试模式或内存使用较高时打印
                    if not self.debug or avg_memory > self.light_cleanup_threshold:
                        print(
                            f"性能统计 - FPS: {self.current_fps:.1f}, 平均处理时间: {avg_processing_time*1000:.1f}ms, "
                            f"最大处理时间: {max_processing_time*1000:.1f}ms, 平均内存: {avg_memory:.1f}MB, "
                            f"队列大小: {self.audio_queue.qsize()}"
                        )
                    self._last_metrics_print = current_time
                
        except Exception as e:
            if self.debug:
                print(f"Performance monitoring error: {e}")
    
    def audio_worker(self):
        """音频处理工作线程 - 优化版本"""
        last_gc_time = time.time()
        last_memory_check = time.time()
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # 优化：减少空轮询时间，提高响应性
                try:
                    audio_data, sample_rate = self.audio_queue.get(timeout=0.01)  # 减少到10ms
                    has_audio = True
                    # 记录最近有音频的时间，并切换到audio状态
                    self._last_audio_event_time = time.time()
                    with self.state_lock:
                        self._maybe_switch_state("audio")
                    # 如果刚经历了静默，则重置音频模型隐状态以避免漂移
                    if getattr(self, 'was_silent', False):
                        try:
                            if hasattr(self.audioModel, 'reset'):
                                self.audioModel.reset()
                        except Exception as e:
                            if self.debug:
                                print(f"音频模型重置失败: {e}")
                        self.was_silent = False
                except queue.Empty:
                    has_audio = False
                
                if has_audio:
                    if self.debug:
                        print(f"开始处理音频数据 - 采样率: {sample_rate}Hz, 长度: {len(audio_data)} samples")
                    
                    # 优化：重采样到目标采样率
                    if sample_rate != self.sample_rate:
                        if self.debug:
                            print(f"音频采样率不匹配，重采样 {sample_rate} → {self.sample_rate}")
                        audio_data = self._resample_audio(audio_data, sample_rate, self.sample_rate)
                    
                    # 分块处理音频
                    chunk_count = 0
                    for i in range(0, len(audio_data), self.samples_per_read):
                        if not self.is_running:
                            break
                        
                        chunk = audio_data[i:i + self.samples_per_read]
                        mouth_frame = self.process_audio_chunk(chunk)
                        chunk_count += 1
                        
                        if mouth_frame is not None:
                            frame = self.render_frame(mouth_frame)
                            
                            if frame is not None and self.virtual_cam is not None:
                                # 发送帧到虚拟摄像头（带过渡）
                                with self.frame_lock:
                                    self.send_with_transition(frame)
                                # 立即释放帧引用
                                del frame
                            
                            # 清理中间变量
                            del mouth_frame
                        
                        # 清理音频块引用
                        del chunk
                        

                    
                    # 清理音频数据引用
                    del audio_data
                    

                    
                    if self.debug:
                        print(f"音频处理完成，共处理 {chunk_count} 个音频块")
                        
                else:
                    # 没有音频：根据静默时间决定是否进入idle
                    now = time.time()
                    if now - self._last_audio_event_time < self.silence_timeout:
                        # 短暂持帧（避免马上切回idle导致跳变）
                        if self._last_output_frame is not None and self.virtual_cam is not None:
                            with self.frame_lock:
                                self.send_with_transition(self._last_output_frame)
                        else:
                            # 尝试播放空闲视频（若可用），否则渲染默认帧
                            if self.virtual_cam is not None:
                                frame_to_send = None
                                if self._idle_video_enabled:
                                    idle_frame = self._read_idle_video_frame()
                                    if idle_frame is not None:
                                        frame_to_send = idle_frame
                                        
                                if frame_to_send is not None:
                                    with self.frame_lock:
                                        self.send_with_transition(frame_to_send)
                                    del frame_to_send
                    else:
                        # 切换到idle并播放idle视频
                        # 标记静默状态以便下次有音频时重置模型
                        if time.time() - self._last_audio_event_time > self.silence_timeout:
                            self.was_silent = True
                        with self.state_lock:
                            self._maybe_switch_state("idle")
                        if self.virtual_cam is not None:
                            frame_to_send = None
                            if self._idle_video_enabled:
                                idle_frame = self._read_idle_video_frame()
                                if idle_frame is not None:
                                    frame_to_send = idle_frame
                            if frame_to_send is not None:
                                with self.frame_lock:
                                    self.send_with_transition(frame_to_send)
                                del frame_to_send
                
                # 记录处理时间
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # 更新性能监控
                self.update_performance_metrics()
                
                # 优化：内存管理 - 更激进版本
                current_time = time.time()
                
                # 定期检查内存使用 - 更激进版本
                if current_time - last_memory_check > self.memory_check_interval:
                    try:
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        
                        # 更激进的分级清理策略
                        if memory_mb > self.force_cleanup_threshold:  # 超过1.5GB
                            print(f"严重警告：内存使用过高 {memory_mb:.1f}MB, 执行深度清理...")
                            self._force_memory_cleanup()
                            # 立即再次检查效果
                            time.sleep(1.0)  # 给系统更多时间稳定
                            after_memory = psutil.Process().memory_info().rss / 1024 / 1024
                            released = memory_mb - after_memory
                            print(f"深度清理后内存: {after_memory:.1f}MB, 释放: {released:.1f}MB")
                            
                            # 如果清理效果不佳，重置AI模型
                            if released < 50:  # 如果释放少于50MB
                                print("清理效果不佳，重置AI模型状态...")
                                try:
                                    if hasattr(self.audioModel, 'reset'):
                                        self.audioModel.reset()
                                    # 强制清理所有PyTorch缓存
                                    import torch
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        torch.cuda.ipc_collect()
                                        torch.cuda.reset_peak_memory_stats()
                                except Exception as e:
                                    print(f"模型重置错误: {e}")
                                    
                        elif memory_mb > self.regular_cleanup_threshold:  # 超过1.2GB
                            print(f"警告：内存使用较高 {memory_mb:.1f}MB, 执行常规清理...")
                            self._cleanup_cached_frames()
                            # 加强GPU清理
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                            except ImportError:
                                pass
                            # 加强垃圾回收
                            for _ in range(5):
                                gc.collect()
                                time.sleep(0.02)
                                
                        elif memory_mb > self.light_cleanup_threshold:  # 超过1.0GB
                            # 轻度清理 + GPU清理
                            gc.collect()
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except ImportError:
                                pass
                            if not self.debug:
                                print(f"轻度清理: 内存 {memory_mb:.1f}MB")
                            
                    except Exception as e:
                        if self.debug:
                            print(f"内存检查错误: {e}")
                    last_memory_check = current_time
                
                # 定期垃圾回收
                if current_time - last_gc_time > self.gc_interval:
                    gc.collect()
                    last_gc_time = current_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio worker error: {e}")
                import traceback
                print(f"Error traceback: {traceback.format_exc()}")
                time.sleep(0.1)

    def start_system(self):
        """启动系统"""
        if self.is_running:
            return False, "System is already running"
        
        # 初始化虚拟摄像头
        if not self.init_virtual_camera():
            return False, "Failed to initialize virtual camera"
        
        self.is_running = True
        
        # 启动音频处理线程
        self.audio_thread = threading.Thread(target=self.audio_worker, daemon=True)
        self.audio_thread.start()
        
        print("DH-LIVE system started successfully")
        return True, "System started"
    
    def stop_system(self):
        """停止系统 - 优化版本"""
        self.is_running = False
        
        # 等待线程结束
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=2)
        
        # 关闭虚拟摄像头
        if self.virtual_cam is not None:
            try:
                self.virtual_cam.close()
            except:
                pass
            self.virtual_cam = None
        
        # 释放空闲视频资源
        if self.idle_video_cap is not None:
            try:
                self.idle_video_cap.release()
            except Exception:
                pass
            self.idle_video_cap = None
            self._idle_video_enabled = False
        
        # 清空音频队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # 清理所有缓存
        self._cleanup_cached_frames()
        self._cached_idle_frame = None
        self._cached_idle_mouth = None
        self._last_output_frame = None
        self._transition_from = None
        
        # 清理预分配缓冲区
        self._preallocated_frames.clear()
        
        # 清理帧池
        with self.frame_pool_lock:
            self.frame_pool.clear()
        
        # 强制垃圾回收
        gc.collect()
        
        print("DH-LIVE system stopped and cleaned up")
        return True, "System stopped"
    
    def process_received_audio(self, audio_data, sample_rate=16000):
        """处理接收到的音频数据 - 优化版本"""
        try:
            # 优化：更激进的队列管理
            while self.audio_queue.qsize() >= self.max_queue_size:
                try:
                    # 主动丢弃最旧的数据防止堆积
                    self.audio_queue.get_nowait()
                    if not self.debug:
                        print(f"警告：音频队列已满({self.max_queue_size})，丢弃最旧数据")
                except queue.Empty:
                    break
            
            # 将音频数据添加到队列进行处理
            self.audio_queue.put((audio_data, sample_rate), block=False)
            
            # 记录最近音频事件时间，提前切到audio状态（减少切换延迟）
            self._last_audio_event_time = time.time()
            with self.state_lock:
                self._maybe_switch_state("audio")
            
            if not self.debug:  # 只在非调试模式显示基本信息
                print(f"音频数据已加入队列，当前队列大小: {self.audio_queue.qsize()}")
            return True
            
        except queue.Full:
            print(f"错误：音频队列已满，无法添加新数据")
            return False
        except Exception as e:
            print(f"音频处理失败: {e}")
            return False
    
    def get_status(self):
        """获取系统状态"""
        try:
            # 获取内存使用情况
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            # 计算平均处理时间
            avg_processing_time = 0
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            return {
                "running": self.is_running,
                "virtual_camera": self.virtual_cam is not None,
                "audio_queue_size": self.audio_queue.qsize(),
                "max_queue_size": self.max_queue_size,
                "character": self.character,
                "performance": {
                    "fps": round(self.current_fps, 1),
                    "memory_mb": round(memory_mb, 1),
                    "cpu_percent": round(cpu_percent, 1),
                    "avg_processing_time_ms": round(avg_processing_time * 1000, 1),
                    "total_frames_processed": len(self.processing_times)
                }
            }
        except Exception as e:
            print(f"Status error: {e}")
            return {
                "running": self.is_running,
                "virtual_camera": self.virtual_cam is not None,
                "audio_queue_size": self.audio_queue.qsize(),
                "character": self.character,
                "error": str(e)
            }

# 全局实例（将在main函数中初始化）
dh_live = None

# FastAPI应用
app = FastAPI(title="DH-LIVE Realtime API")



# 系统自动启动，无需手动启动/停止API

@app.post("/receive_audio")
async def receive_audio_api(audio_file: UploadFile = File(...)):
    """接收音频文件API"""
    try:
        print(f"接收到音频文件: {audio_file.filename}, 大小: {audio_file.size} bytes")
        
        # 读取上传的音频文件
        audio_content = await audio_file.read()
        print(f"音频内容读取完成，长度: {len(audio_content)} bytes")
        
        # 解析音频数据
        audio_data, sample_rate = sf.read(BytesIO(audio_content))
        print(f"音频解析完成 - 采样率: {sample_rate}Hz, 数据长度: {len(audio_data)} samples, 时长: {len(audio_data)/sample_rate:.2f}s")
        
        # 处理音频数据
        success = dh_live.process_received_audio(audio_data, sample_rate)
        
        if success:
            print("音频已成功加入处理队列")
            return JSONResponse(content={"status": "success", "message": "Audio received and queued for processing", "audio_info": {"sample_rate": sample_rate, "duration": len(audio_data)/sample_rate, "samples": len(audio_data)}})
        else:
            print("音频处理失败")
            raise HTTPException(status_code=500, detail="Failed to process audio")
            
    except Exception as e:
        print(f"音频处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

@app.post("/receive_audio_base64")
async def receive_audio_base64_api(request: dict):
    """接收Base64编码音频数据API"""
    try:
        audio_base64 = request.get("audio_data", "")
        sample_rate = request.get("sample_rate", 16000)
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="Audio data is required")
        
        # 解码Base64音频数据
        audio_bytes = base64.b64decode(audio_base64)
        
        # 解析音频数据
        audio_data, actual_sample_rate = sf.read(BytesIO(audio_bytes))
        
        # 使用实际采样率或传入的采样率
        final_sample_rate = actual_sample_rate if actual_sample_rate else sample_rate
        
        # 处理音频数据
        success = dh_live.process_received_audio(audio_data, final_sample_rate)
        
        if success:
            return JSONResponse(content={"status": "success", "message": "Audio received and queued for processing"})
        else:
            raise HTTPException(status_code=500, detail="Failed to process audio")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

@app.get("/status")
async def get_status():
    """获取系统状态"""
    status = dh_live.get_status()
    return JSONResponse(status)

@app.post("/cleanup_memory")
async def cleanup_memory_api():
    """手动触发内存清理"""
    try:
        # 获取清理前的内存使用
        process = psutil.Process()
        before_mb = process.memory_info().rss / 1024 / 1024
        
        # 执行强制清理
        dh_live._force_memory_cleanup()
        
        # 等待一下让系统稳定
        time.sleep(1)
        
        # 获取清理后的内存使用
        after_mb = process.memory_info().rss / 1024 / 1024
        freed_mb = before_mb - after_mb
        
        return JSONResponse({
            "status": "success",
            "message": "Memory cleanup completed",
            "memory_before_mb": round(before_mb, 1),
            "memory_after_mb": round(after_mb, 1),
            "memory_freed_mb": round(freed_mb, 1)
        })
        
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Memory cleanup failed: {str(e)}"
        }, status_code=500)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DH-LIVE Realtime System")
    parser.add_argument("--audio_model", 
                       default="checkpoint/audio.pkl", 
                       help="音频模型路径 (默认: checkpoint/audio.pkl)")
    parser.add_argument("--render_model", 
                       default="checkpoint/15000.pth", 
                       help="渲染模型路径 (默认: checkpoint/15000.pth)")
    parser.add_argument("--character", 
                       default="dw", 
                       help="角色名称 (默认: dw)")
    parser.add_argument("--host", 
                       default="0.0.0.0", 
                       help="服务器主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", 
                       type=int, 
                       default=8000, 
                       help="服务器端口 (默认: 8000)")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    print(f"启动DH-LIVE系统...")
    print(f"音频模型: {args.audio_model}")
    print(f"渲染模型: {args.render_model}")
    print(f"角色: {args.character}")
    print(f"服务地址: {args.host}:{args.port}")
    
    # 创建DH-LIVE实例（自动启动系统）
    # dh_live已在全局定义，无需再次声明global
    dh_live = DHLiveRealtime(
        audio_model_path=args.audio_model,
        render_model_path=args.render_model,
        character=args.character,
        auto_start=True
    )
    
    print("DH-LIVE系统已自动启动，等待音频输入...")
    print(f"API服务启动在: http://{args.host}:{args.port}")
    print("可用的API端点:")
    print("  POST /receive_audio - 接收音频文件")
    print("  POST /receive_audio_base64 - 接收Base64编码音频")
    print("  GET /status - 获取系统状态")
    
    # 启动FastAPI服务器
    uvicorn.run(app, host=args.host, port=args.port)