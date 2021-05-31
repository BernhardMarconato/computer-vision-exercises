from typing import List, Tuple
import cv2 as cv
import numpy as np


class CameraCalibrator:
    def __init__(self, frames: List[np.array], chk_size: Tuple[int, int]):
        self.frames = frames
        self.chk_size = chk_size
        self.obj_points = []
        self.img_points = []

    def detect_checkerboard_frame_ids(self) -> List[int]:
        found_ids = []
        self.detected_checkerboards = []
        for frame_id, frame in enumerate(self.frames):
            gray = frame#cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, _ = cv.findChessboardCorners(gray, self.chk_size, cv.CALIB_CB_ADAPTIVE_THRESH)
            if ret:
                found_ids.append(frame_id)
                print("FOUND " + str(frame_id))
        return found_ids

    def detect_checkerboard_frames(self) -> List[np.array]:
        frame_ids = self.detect_checkerboard_frame_ids()
        return self.__get_frames(frame_ids)

    def __get_frames(self, frame_ids: List[int]):
        return [self.frames[frame_id] for frame_id in frame_ids]

    def get_camera_params(self, frame_ids: List[int] = None, frames: List[np.array] = None, show_img: bool = False) -> Tuple:
        if not frames:
            if self.frames:
                frames = self.frames
            else:
                frames = self.__get_frames(frame_ids)

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        ch_rows = self.chk_size[0]
        ch_cols = self.chk_size[1]
        objp = np.zeros((ch_cols * ch_rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:ch_rows, 0:ch_cols].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane.

        for frame in frames:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.chk_size, 0)
            # If found, add object points, image points (after refining them)
            if ret:
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.obj_points.append(objp)
                self.img_points.append(corners2)
                if show_img:
                    fcopy = frame.copy()
                    cv.drawChessboardCorners(fcopy, self.chk_size, corners2, ret)
                    cv.imshow('img', fcopy)
                    cv.waitKey()
            else:
                print("ERR")
        cv.destroyAllWindows()

        return cv.calibrateCamera(self.obj_points, self.img_points, gray.shape[::-1], None, None)

    def undistort(self, image: np.array, filename: str, params: Tuple):
        ret, mtx, dist, rvecs, tvecs = params

        h, w = image.shape[:2]
        cv.imwrite(f'{filename}_0.png', image)

        # undistort
        dst = cv.undistort(image, mtx, dist, None, None)
        cv.imwrite(f'{filename}_1.png', dst)

        # crop the image
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        dst = cv.undistort(image, mtx, dist, None, newcameramtx)
        cv.imwrite(f'{filename}_2.png', dst)

        # crop the image
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.5, (w, h))
        dst = cv.undistort(image, mtx, dist, None, newcameramtx)
        cv.imwrite(f'{filename}_3.png', dst)

        # crop the image
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv.undistort(image, mtx, dist, None, newcameramtx)
        cv.imwrite(f'{filename}_4.png', dst)

        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv.imwrite(f'{filename}_5.png', dst)





