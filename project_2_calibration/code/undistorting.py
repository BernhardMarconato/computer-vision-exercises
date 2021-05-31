import os
import shutil
import array_to_latex as a2l
import numpy as np

from CameraCalibrator import CameraCalibrator
from VideoFrameExtractor import VideoFrameExtractor

chc1 = "videos/ch0.mp4"  # "videos/checkerboard_000.h264"
chc2 = "videos/ch1.mp4"  # "videos/checkerboard_019.h264"
chc_size = (7, 6)

extractor_1 = VideoFrameExtractor(chc1)
extractor_2 = VideoFrameExtractor(chc2)

fids_1 = [1616, 848, 1347, 938, 1343, 613, 925, 963, 1617, 1367]
fids_2 = [133, 136, 132, 129, 176, 148, 179, 147, 126, 135]

frames_1 = extractor_1.get_video_frames(fids_1)
frames_2 = extractor_2.get_video_frames(fids_2)
m_frames = frames_1 + frames_2

# get camera parameters
calibrator = CameraCalibrator(m_frames, chc_size)
params = calibrator.get_camera_params()
print(params)

print(params[3])

# undistort images
subfolder = "calibration"
shutil.rmtree(subfolder, ignore_errors=True)
os.mkdir(subfolder)
for fid, frame in enumerate(m_frames):
    calibrator.undistort(frame, f"{subfolder}/img{fid}", params)