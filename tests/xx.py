import numpy as np
import PyNvCodec as nvc
import nvjpeg
import os, uuid, cv2
from fvcore.common.timer import Timer

def xxxxxx():
    gpuID, gpuId = 0, 0
    model_w = 640
    model_h = 384
    model_c = 3
    nvDec = nvc.PyNvDecoder('ch03_20200520134829_cutted.mp4', gpuID)
    nvCvt = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvDec.Format(), nvc.PixelFormat.YUV420, gpuId)
    nvRes = nvc.PySurfaceResizer(model_w, model_h, nvCvt.Format(), gpuId)
    to_rgb = nvc.PySurfaceConverter(model_w, model_h, nvRes.Format(), nvc.PixelFormat.RGB, gpuId)
    nvDwn = nvc.PySurfaceDownloader(model_w, model_h, to_rgb.Format(), gpuID)

    rawFrame = np.ndarray(shape=(model_w * model_h * model_c), dtype=np.uint8)
    xx = nvDwn.DownloadSingleSurface(to_rgb.Execute(nvRes.Execute(nvCvt.Execute(nvDec.DecodeSingleSurface()))),
                                     rawFrame)
    if xx:
        frame = rawFrame.reshape(model_h, model_w, model_c)[..., ::-1]

        nj = nvjpeg.NvJpeg()
        tt = Timer()
        for _ in range(99):
            with open(os.path.join('/dev/shm/', f"{uuid.uuid4().hex}.jpg"), "wb") as fid:
                frame_jpg = nj.encode(frame)
                fid.write(frame_jpg)
        print(f"nvjpeg time is {tt.seconds()}")
        tt.reset()
        for _ in range(99):
            cv2.imwrite(os.path.join('/dev/shm/', f"{uuid.uuid4().hex}.jpg"), frame)
        print(f"cv2 time is {tt.seconds()}")
    else:
        raise RuntimeError


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            # because I wanted to initialise my class
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]

class nvjpegencoder(metaclass=Singleton):
    def __init__(self):
        self.nj = nvjpeg.NvJpeg()
    def doencode(self,frame):
        return self.nj.encode(frame)




# def test_perf(img:np.ndarray):

