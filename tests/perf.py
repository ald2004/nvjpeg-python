#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2
import glob
from fvcore.common.timer import Timer
import uuid
for lib in glob.glob(os.path.join(os.path.dirname(__file__), "../build/lib.*")):
    sys.path.append(lib)

from nvjpeg import NvJpeg
runtimes=999
nj = NvJpeg()


image_file = os.path.join(os.path.dirname(__file__), "04.jpg")
cv_np = cv2.imread(image_file)
# print(cv_np.shape)

print(f"cv2 write...")
tt=Timer()
for _ in range(runtimes):
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "out", f"python-opencv-test-{uuid.uuid4().hex}.jpg"), cv_np)
tt.pause()
print(f"cv2 write time is : {tt.seconds()}")

print(f"nvJpeg write ...")
tt.reset()
for _ in range(runtimes):
    nj_jpg = nj.encode(cv_np)
    fp = open(os.path.join(os.path.dirname(__file__), "out", f"python-nvJpeg-test-{uuid.uuid4().hex}.jpg"), "wb")
    fp.write(nj_jpg)
    fp.close()

tt.pause()
print(f"nvJpeg write time is : {tt.seconds()}")