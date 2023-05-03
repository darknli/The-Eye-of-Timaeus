import cv2
import numpy as np

from .base import Eye, ncolors
from typing import Optional, Union


class CVTracker:
    """
    使用opencv-contrib-python库自带的一些跟踪方法
    """
    cv2_tracker = {
        "csrt": cv2.legacy.TrackerCSRT_create,
        "kcf": cv2.legacy.TrackerKCF_create,
        "boosting": cv2.legacy.TrackerBoosting_create,
        "mil": cv2.legacy.TrackerMIL_create,
        "tld": cv2.legacy.TrackerTLD_create,
        "medianflow": cv2.legacy.TrackerMedianFlow_create,
        "mosse": cv2.legacy.TrackerMOSSE_create
    }

    def __init__(self, tracker_type: str, scale=0.2):
        if tracker_type in self.cv2_tracker:
            self.tracker_type = self.cv2_tracker[tracker_type]
        else:
            raise NotImplementedError
        self.trackers = []
        self.cls = []
        self.scale = scale

    def track_objs(self, image, boxes):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, None, fx=self.scale, fy=self.scale)
        self.trackers = []
        self.cls = []
        for b in boxes.astype(int).tolist():
            self.cls.append(b[0])
            tracker = self.tracker_type()
            init_box = (b[2], b[3], b[4] - b[2], b[5] - b[3])
            assert init_box[2] > 0 and init_box[3] > 0
            init_box = tuple([b * self.scale for b in init_box])
            tracker.init(image, init_box)
            self.trackers.append(tracker)

    def __call__(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, None, fx=self.scale, fy=self.scale)
        all_boxes = []
        for trk, c in zip(self.trackers, self.cls):
            status, box = trk.update(image)
            if len(box) != 4 or box[2] == 0 or box[3] == 0:
                continue
            box = [b / self.scale for b in box]
            new_box = (c, 1, box[0], box[1], box[0] + box[2], box[1] + box[3])
            all_boxes.append(new_box)
        all_boxes = np.array(all_boxes)
        return all_boxes


class TrackEye(Eye):
    """
    对视频流做目标检测工程化操作

    Parameters
    ----------
    model : any. 模型推理的类，要求它
        * 已实现了__call__且输入是BGR的图片，输出是np.ndarray格式shape=(N, 6)的boxes其中最后一维是cls_type,conf,x1,y1,x2,y2。
        * 已实现了track_objs方法来设置初始框，输入分别是image和boxes，boxes要求依然是shape=(N, 6)
    video_type : Optional[Union[str, int]]. 目前支持摄像头、本地视频和单帧图片三种模型
        * int. 摄像头的序号
        * `other`. 本地视频
        * None. 单帧图片（暂不支持）
    display_name : Optional[str]. 显示窗口的名称，如果是None则不显示窗口
    """

    def __init__(self, model,
                 color_boxes: list,
                 video_type: Optional[Union[str, int]] = None,
                 display_name: Optional[str] = None,
                 video_width: int = 1920,
                 video_height: int = 1080,
                 fps: int = 30,
                 ):
        super().__init__(video_type, display_name, video_width, video_height, fps)
        if isinstance(model, str):
            self.model = CVTracker(model)
        else:
            self.model = model
        self.color_boxes = color_boxes
        self.has_obj = False
        print("Track-Eye初始化完成!")

    def tracking(self, image, init_boxes):
        """

        Parameters
        ----------
        image : np.ndarray. 初始框对应的那张图片
        init_boxes : np.ndarray. 区别于目标检测，跟踪器需要再给出初始框才能执行。
            初始框shape=(N, 6)，最后一维是cls_type,conf,x1,y1,x2,y2
        """
        self.model.track_objs(image, init_boxes)
        self.has_obj = True

    def predict(self, image):
        show_image = image.copy()
        if not self.has_obj:
            self.show_image = show_image if self.display_name else None
            return np.zeros((0, 6))
        boxes = self.model(show_image)
        if self.display_name:
            for cls, conf, x1, y1, x2, y2 in boxes.tolist():
                cls = int(cls)
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                color = self.color_boxes[int(cls)]
                cv2.rectangle(show_image, (x1, y1), (x2, y2), color, 1)
                title = f"{cls}"
                title_lt = (x1, y1 - 15)
                title_rb = (x1 + len(title) * 10, y1)
                cv2.rectangle(show_image, title_lt, title_rb, color, -1)
                cv2.putText(show_image, title, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            self.show_image = show_image
        return boxes

    @property
    def display_frame(self):
        return self.show_image
