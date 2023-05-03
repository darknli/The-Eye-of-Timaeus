# The Eye of Timaeus
该项目定义了一些piplines，可以针对视频流的一些能力，诸如检测、识别、跟踪、分割等实现工程化，将其和模型推理解耦开来
## 已支持能力
 - [x] 支持多进程检测/跟踪，避免卡顿
 - [x] 对视频流做目标检测
 - [x] 对视频流做目标跟踪
 - [ ] 对视频流做分割
 - [ ] 对视频流做姿态估计
## 使用示例
- 创建一套检测+跟踪的pipline
```commandline
# 引入对应的包
from teot.detect import DetectEye
from teot.track import TrackEye
from teot.multi import MultiEye

# 定义好你的检测器
det = ...
# 将检测器传入DetectEye中创建对象
det_eye = DetectEye(det)

# 创建一个TrackEye对象用于跟踪，这里的跟踪方法是KCF
track_eye = TrackEye("kcf", det_eye.color_boxes)

# 创建一个检测+跟踪的混合pipline，并运行
multi_eye = MultiEye(det_eye, track_eye, video_type="视频路径", display_name="win")
multi_eye.run()
```
下面是运行权游的demo

![图例1](./src/1.png "图例1")
![图例2](./src/2.png "图例2")

## TODO
 - [ ] 解决视频人物动作过快，显示的预测结果和对应帧有明显延迟的问题
