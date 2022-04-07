import cv2
import threading
import logging
import time

from PIL import Image

class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened.")
        
        # 初始化参数
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

        # 获取视频参数
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Camera opened. FPS: {fps}, Resolution: {width}x{height}")

    def _update_frame(self):
        """后台线程，持续抓取帧"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame
            time.sleep(0.01)  # 稍作等待，避免 CPU 占用过高

    def start(self):
        """启动摄像头读取线程"""
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def stop(self):
        """停止摄像头读取"""
        self.running = False
        self.thread.join()
        self.cap.release()

    def get_frame(self):
        """获取最新帧"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
        
    def get_image(self):
        """获取最新帧并转为 PIL.Image 对象"""
        with self.lock:
            if self.frame is None:
                return None
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
