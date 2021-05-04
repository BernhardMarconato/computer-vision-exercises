import numpy as np
import random as rng
import matplotlib.pyplot as plt
import cv2
from enum import Enum, auto


class DocumentFormat(Enum):
    Auto = auto()
    A4 = auto()


class DocumentScanner:
    def __init__(
            self,
            filename: str,
            document_format: DocumentFormat,
            apply_effects: bool):
        self.fileName = filename
        self.document = cv2.imread(self.fileName)
        self.format = document_format
        self.apply_effects = apply_effects
        self.__image_history = []

    IMAGE_HEIGHT = 700

    def scan(self):
        self.__add_to_history("Original", self.document)

        img = self.__prepare_image(self.document)
        self.__add_to_history("Prepared", img)

        edged = self.__get_edged_image(img)

        contour = self.__get_best_contour(edged)
        contour_img = img.copy()
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
        self.__add_to_history("Detected contour", contour_img)

        transformed_img = self.__transform_image(self.document, contour)
        self.__add_to_history("Transformed", transformed_img)

        if self.apply_effects:
            transformed_img = self.__apply_image_effects(transformed_img)

        self.__add_to_history("Final scan", transformed_img)
        return transformed_img

    def __prepare_image(self, img):
        # scale down image
        ratio = self.IMAGE_HEIGHT / img.shape[0]
        img = cv2.resize(img, (int(ratio * img.shape[1]), self.IMAGE_HEIGHT))

        # gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return img

    def __get_edged_image(self, img):
        # run canny edge detector
        edged = cv2.Canny(img, 70, 120)
        self.__add_to_history("Canny Edge", edged)

        # connect lines that may be disconnected a bit
        kernel = np.ones((5, 5), np.uint8)
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        self.__add_to_history("Canny Edge Morphed", edged)

        return edged

    def __get_best_contour(self, edged_img):
        # get contours
        contours, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours for visualization
        drawing = np.zeros((edged_img.shape[0], edged_img.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, None, 0)
        self.__add_to_history("Detected Contours", drawing)

        # calculate contour area
        contour_areas = map(cv2.contourArea, contours)
        contours_with_areas = zip(contours, contour_areas)

        # contour must span at least 40% of image area
        img_height, img_width = edged_img.shape
        min_area = img_height * img_width * 0.4
        filtered_contours = filter(lambda tup: tup[1] >= min_area, contours_with_areas)

        # sort remaining contours by the area they span
        sorted_contours = sorted(filtered_contours, key=lambda tup: tup[1], reverse=True)

        for contour, _ in sorted_contours:
            # approximate the contour as rectangle
            eps = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)
            # use first (biggest) contour that has 4 corners as our best contour
            if len(approx) == 4:
                return approx

        # if no fitting contour was found, return contour of entire image
        return np.array([[[0, 0]], [[0, img_height]], [[img_width, img_height]], [[img_width, 0]]])

    def __transform_image(self, img, contour):
        contour_pts = self.__order_points_clockwise(contour.reshape(4, 2))
        scale = img.shape[0] / self.IMAGE_HEIGHT
        src_pts = (contour_pts * scale).astype(np.float32)

        width = max(np.linalg.norm(src_pts[3] - src_pts[2]),
                    np.linalg.norm(src_pts[1] - src_pts[0]))

        if self.format == DocumentFormat.A4:
            height = width * np.sqrt(2)
        else:
            height = max(np.linalg.norm(src_pts[3] - src_pts[0]),
                         np.linalg.norm(src_pts[2] - src_pts[1]))

        dst_pts = np.array([[0, 0],
                            [width, 0],
                            [width, height],
                            [0, height]],
                           dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(img, M, (int(width), int(height)))

    def __apply_image_effects(self, img):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 15)

    def __apply_image_effects_ex(self, img):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thrimg = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)
        return thrimg

    def __order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum
        # the bottom-right point will have the largest sum
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # the top-right point will have the smallest difference
        # the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def __add_to_history(self, title, img):
        self.__image_history.append((title, img))

    def save_image(self, img, destFilePath):
        cv2.imwrite(destFilePath, img)

    def show_image_history(self):
        rows = int(np.sqrt(len(self.__image_history)))
        cols = int(np.ceil(len(self.__image_history) / rows))

        fig, axes = plt.subplots(rows, cols)
        fig.tight_layout()
        i = 0
        for title, img in self.__image_history:
            axes.ravel()[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes.ravel()[i].set_title(title)
            axes.ravel()[i].axis("off")
            i = i + 1

        while i < rows * cols:
            axes.ravel()[i].axis("off")
            i = i + 1

        plt.show()
