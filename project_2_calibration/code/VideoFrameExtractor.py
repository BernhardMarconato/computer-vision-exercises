from typing import List
import cv2


class VideoFrameExtractor:
    def __init__(self, filename: str):
        self.filename = filename
        self.__frame_count_cache = None

    def get_video_frame_count(self):
        if self.__frame_count_cache:
            return self.__frame_count_cache

        cap = cv2.VideoCapture(self.filename)

        # doesnt work with given videos
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count >= 0:
            return frame_count

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        cap.release()

        self.__frame_count_cache = frame_count
        return frame_count

    def __get_video_frames_fallback(self, frame_ids: List[int]):
        cap = cv2.VideoCapture(self.filename)
        try:
            frames = []
            frame_id = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id in map(int, frame_ids):
                    frames.append(frame)
                frame_id += 1
        finally:
            cap.release()

        return frames

    def get_video_frames(self, frame_ids: List[int]):
        frames = []
        cap = cv2.VideoCapture(self.filename)
        try:
            for frame_id in map(int, frame_ids):
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
                    return self.__get_video_frames_fallback(frame_ids)
                ret, frame = cap.read()
                if not ret:
                    return self.__get_video_frames_fallback(frame_ids)
                frames.append(frame)
        finally:
            cap.release()

        return frames

    def get_video_frames_generator(self):
        cap = cv2.VideoCapture(self.filename)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()
