from xx import nvjpegencoder
import numpy as np
import PyNvCodec as nvc
import os, uuid, cv2
from fvcore.common.timer import Timer
import fire
from logger import setup_logger
# import nvjpeg

def test_perf():
    logger = setup_logger(name='nvxx')
    gpuID, gpuId = 0, 0
    model_w = 640
    model_h = 384
    model_w = 1920
    model_h = 1080
    model_w = 2560
    model_h = 1440
    model_c = 3
    runtimes = 99
    nvDec = nvc.PyNvDecoder('ch03_20200520134829_cutted.mp4', gpuID)
    nvCvt = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvDec.Format(), nvc.PixelFormat.YUV420, gpuId)
    nvRes = nvc.PySurfaceResizer(model_w, model_h, nvCvt.Format(), gpuId)
    to_rgb = nvc.PySurfaceConverter(model_w, model_h, nvRes.Format(), nvc.PixelFormat.RGB, gpuId)
    nvDwn = nvc.PySurfaceDownloader(model_w, model_h, to_rgb.Format(), gpuID)

    rawFrame = np.ndarray(shape=(model_w * model_h * model_c), dtype=np.uint8)
    xx = nvDwn.DownloadSingleSurface(to_rgb.Execute(nvRes.Execute(nvCvt.Execute(nvDec.DecodeSingleSurface()))),
                                     rawFrame)

    if xx:
        # nj = nvjpeg.NvJpeg()
        nj = nvjpegencoder()
        tt = Timer()
        for _ in range(runtimes):
            xx = nvDwn.DownloadSingleSurface(to_rgb.Execute(nvRes.Execute(nvCvt.Execute(nvDec.DecodeSingleSurface()))),
                                             rawFrame)
            if not xx:
                raise RuntimeError
            frame = rawFrame.reshape(model_h, model_w, model_c)[..., ::-1]
            # frame04 = cv2.imread('04.jpg')
            with open(os.path.join('/dev/shm/nvxx/out', f"{uuid.uuid4().hex}_nvjpeg.jpg"), "wb") as fid:
                frame_jpg = nj.doencode(frame)
                # frame_jpg = nj.doencode(frame04)
                # frame_jpg = nj.encode(frame)
                fid.write(frame_jpg)

        # logger.debug(f"nvjpeg time is {tt.seconds()} and shape is {frame04.shape}")
        logger.debug(f"nvjpeg time is {tt.seconds()} and shape is {frame.shape}")
        tt.reset()
        for _ in range(runtimes):
            # frame04 = cv2.imread('04.jpg')
            xx = nvDwn.DownloadSingleSurface(to_rgb.Execute(nvRes.Execute(nvCvt.Execute(nvDec.DecodeSingleSurface()))),
                                             rawFrame)
            if not xx:
                raise RuntimeError
            frame = rawFrame.reshape(model_h, model_w, model_c)[..., ::-1]
            cv2.imwrite(os.path.join('/dev/shm/nvxx/out', f"{uuid.uuid4().hex}_cv2.jpg"), frame)
            # cv2.imwrite(os.path.join('/dev/shm/nvxx/out', f"{uuid.uuid4().hex}_cv2.jpg"), frame04)

        # logger.debug(f"cv2 time is {tt.seconds()}and shape is {frame04.shape}")
        logger.debug(f"cv2 time is {tt.seconds()}and shape is {frame.shape}")
    else:
        raise RuntimeError

if __name__=="__main__":
    fire.Fire()
