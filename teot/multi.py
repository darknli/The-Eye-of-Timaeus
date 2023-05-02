import time
from typing import Optional, Union
from .track import Eye
from .track import TrackEye
from .detect import DetectEye
import threading
from .thread_utils import DetectThread, TrackThread
import cv2





class MultiEye(Eye):
    """
    对视频流做一些工程化操作

    Parameters
    ----------
    detect_eye : Optional[DetectEye]
    track_eye : Optional[TrackEye]
    detect_interval : int, default 30. 检测间隔，单位是ms
    video_type : Optional[Union[str, int]]. 目前支持摄像头、本地视频和单帧图片三种模型
        * int. 摄像头的序号
        * `other`. 本地视频
        * None. 单帧图片（暂不支持）
    display_name : Optional[str]. 显示窗口的名称，如果是None则不显示窗口
    """
    def __init__(self,
                 detect_eye: Optional[DetectEye] = None,
                 track_eye: Optional[TrackEye] = None,
                 detect_interval: int = 30,
                 video_type: Optional[Union[str, int]] = None,
                 display_name: Optional[str] = None,
                 video_width: int = 1920,
                 video_height: int = 1080,
                 fps: int = 30,
                 ):
        super().__init__(video_type, display_name, video_width, video_height, fps)
        if track_eye and not detect_eye:
            raise ValueError("跟踪必须要有检测")
        assert detect_eye
        self.detect_lock = threading.Lock() if detect_eye else None
        self.detect_interval = detect_interval / 1000
        self.latest_time = time.time()
        self.data = {}
        self.mtx_box = threading.Lock()
        self.mtx_image = threading.Lock()
        self.detect_thread = DetectThread(detect_eye, self.data, self.mtx_box, self.mtx_image, detect_interval)
        self.track_thread = TrackThread(track_eye, self.data, self.mtx_box, self.mtx_image) if track_eye else None

        if self.detect_thread:
            self.detect_thread.start()
        if self.track_thread:
            self.track_thread.start()

    def predict(self, image):
        self.mtx_image.acquire()
        self.data["image"] = image
        self.mtx_image.release()
        self.mtx_box.acquire()
        if "boxes" in self.data:
            print(self.data["boxes"])
            boxes = self.data["boxes"].tolist()
        else:
            boxes = []
        self.mtx_box.release()
        show_image = image.copy()
        for cls, conf, x1, y1, x2, y2 in boxes:
            cls = int(cls)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            color = self.detect_thread.model.color_boxes[int(cls)]
            cv2.rectangle(show_image, (x1, y1), (x2, y2), color, 1)
            title = f"{cls}:{round(conf, 2)}"
            title_lt = (x1, y1 - 15)
            title_rb = (x1 + len(title) * 10, y1)
            cv2.rectangle(show_image, title_lt, title_rb, color, -1)
            cv2.putText(show_image, title, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
        self.show_image = show_image

    @property
    def display_frame(self):
        return self.show_image

