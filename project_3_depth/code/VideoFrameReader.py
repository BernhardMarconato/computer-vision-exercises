import glob
from typing import List
import cv2


class VideoFrameReader:
    def __init__(self, directory: str):
        self.directory = directory

    def get_video_frames(self, frame_ids: List[int], search_string: str = "*.png"):
        frames = []
        for fid in frame_ids:
            p = self.directory + search_string.replace("*", str(fid))
            f = cv2.imread(p)
            frames.append(f)

        return frames

    def get_video_frames_generator(self, search_string: str = "*.png"):
        images = sorted(glob.glob(self.directory + search_string), key=len)
        for image in images:
            yield cv2.imread(image)
