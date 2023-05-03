import abc
import warnings
from typing import Optional, Union
import cv2
import colorsys
import random
import time
import numpy as np


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    """获取区分度比较高的num个BGR颜色，代码来自https://github.com/choumin/ncolors"""
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([b, g, r])

    return rgb_colors


class Eye:
    """
    对视频流做一些工程化操作

    Parameters
    ----------
    video_type : Optional[Union[str, int]]. 目前支持摄像头、本地视频和单帧图片三种模型
        * int. 摄像头的序号
        * `other`. 本地视频
        * None. 单帧图片（暂不支持）
    display_name : Optional[str]. 显示窗口的名称，如果是None则不显示窗口
    """

    def __init__(self, video_type: Optional[Union[str, int]] = None,
                 display_name: Optional[str] = None,
                 video_width: int = 1920,
                 video_height: int = 1080,
                 fps: int = 30,
                 ):
        self.flip = False
        if isinstance(video_type, int) or isinstance(video_type, str):
            self.flip = isinstance(video_type, int)
            self.cap = cv2.VideoCapture(video_type)
            self.cap.set(3, video_width)  # 设置分辨率
            self.cap.set(4, video_height)
            print("原视频帧率是", int(self.cap.get(cv2.CAP_PROP_FPS)))
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps
            print("现视频帧率是", int(self.cap.get(cv2.CAP_PROP_FPS)))
        else:
            self.cap = None
        self.display_name = display_name
        self.latest_time = time.time()

    def run(self, ):
        interval = 1 / self.fps

        while True:
            cur_time = time.time()
            # 控制播放帧率
            if cur_time - self.latest_time < interval:
                continue
            frame = self.next_frame()
            if frame is None:
                print("done")
                break
            self.predict(frame)
            self.show()
            self.latest_time = cur_time

    def next_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if ret:
            if self.flip:
                return np.ascontiguousarray(frame[:, ::-1])
            return frame
        return None

    def show(self):
        frame = self.display_frame
        if frame is not None and self.display_name:
            cv2.imshow(self.display_name, frame)
            cv2.waitKey(1)
        elif self.display_name:
            warnings.warn("该帧没有内容，是否需要在初始化的时候关闭显示")

    @property
    def display_frame(self):
        return None

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        ...
