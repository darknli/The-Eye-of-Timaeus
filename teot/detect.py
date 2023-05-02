import cv2
from .base import Eye, ncolors
from typing import Optional, Union


class DetectEye(Eye):
    """
    对视频流做目标检测工程化操作

    Parameters
    ----------
    model : any. 模型推理的类，要求它已实现了__call__且输入是BGR的图片，
        输出是np.ndarray格式shape=(N, 6)的boxes其中最后一维是cls_type,conf,x1,y1,x2,y2。
        此外，model.num_classes需要给出可预测的类别总数
    video_type : Optional[Union[str, int]]. 目前支持摄像头、本地视频和单帧图片三种模型
        * int. 摄像头的序号
        * `other`. 本地视频
        * None. 单帧图片（暂不支持）
    display_name : Optional[str]. 显示窗口的名称，如果是None则不显示窗口
    """

    def __init__(self, model,
                 video_type: Optional[Union[str, int]] = None,
                 display_name: Optional[str] = None,
                 video_width: int = 1920,
                 video_height: int = 1080,
                 fps: int = 30,
                 ):
        super().__init__(video_type, display_name, video_width, video_height, fps)
        self.model = model
        self.color_boxes = ncolors(model.num_classes)
        self.show_image = None

    def predict(self, image):
        print("检测一帧")
        boxes = self.model(image)
        if self.display_name:
            show_image = image.copy()
            for cls, conf, x1, y1, x2, y2 in boxes.tolist():
                cls = int(cls)
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                color = self.color_boxes[int(cls)]
                cv2.rectangle(show_image, (x1, y1), (x2, y2), color, 1)
                title = f"{cls}:{round(conf, 2)}"
                title_lt = (x1, y1 - 15)
                title_rb = (x1 + len(title) * 10, y1)
                cv2.rectangle(show_image, title_lt, title_rb, color, -1)
                cv2.putText(show_image, title, (x1, y1), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            self.show_image = show_image
        return boxes

    @property
    def display_frame(self):
        return self.show_image


