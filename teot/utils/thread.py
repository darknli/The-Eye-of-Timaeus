from typing import Union, Optional
import time
import threading


class DetectThread(threading.Thread):
    def __init__(self, eye, data,
                 mutex_box: Optional[threading.Lock] = None,
                 mutex_img: Optional[threading.Lock] = None,
                 interval: int = 100):
        threading.Thread.__init__(self)
        self.eye = eye
        self.mtx_box = mutex_box if mutex_box else threading.Lock()
        self.mtx_img = mutex_img if mutex_img else threading.Lock()
        self.fast_detect = True
        self.data = data
        self.latest_time = time.time()
        self.interval = interval / 1000

    def predict(self):
        self.latest_time = time.time()

    def run(self) -> None:
        while True:
            cur_time = time.time()
            self.mtx_box.acquire()
            self.fast_detect = self.fast_detect or "boxes" not in self.data or len(self.data["boxes"])
            self.mtx_box.release()
            if self.fast_detect and (cur_time - self.latest_time) > self.interval:
                self.mtx_img.acquire()
                if "image" not in self.data:
                    self.mtx_img.release()
                    continue
                image = self.data["image"]
                self.mtx_img.release()
                boxes = self.eye.predict(image)
                self.mtx_box.acquire()
                self.data["boxes"] = boxes.copy()
                self.data["boxes_det"] = boxes.copy()
                self.mtx_box.release()
                self.fast_detect = False
                self.latest_time = time.time()


class TrackThread(threading.Thread):
    def __init__(self, eye, data,
                 mutex_box: Optional[threading.Lock] = None,
                 mutex_img: Optional[threading.Lock] = None,
                 interval: int = 30):
        threading.Thread.__init__(self)
        self.eye = eye
        self.mtx_box = mutex_box if mutex_box else threading.Lock()
        self.mtx_img = mutex_img if mutex_img else threading.Lock()
        self.data = data
        self.latest_time = time.time()
        self.interval = interval / 1000

    def run(self) -> None:
        while True:
            cur_time = time.time()
            if (cur_time - self.latest_time) > self.interval:
                self.mtx_img.acquire()
                if "image" not in self.data:
                    self.mtx_img.release()
                    continue
                image = self.data["image"]
                self.mtx_img.release()
                if "boxes_det" in self.data:
                    self.mtx_box.acquire()
                    self.eye.tracking(image, self.data["boxes_det"])
                    del self.data["boxes_det"]
                    self.mtx_box.release()
                    continue
                if not self.eye.has_obj:
                    continue
                boxes = self.eye.predict(self.data["image"])
                self.mtx_box.acquire()
                self.data["boxes"] = boxes.copy()
                self.data["boxes_track"] = boxes.copy()
                self.mtx_box.release()
                self.latest_time = time.time()
