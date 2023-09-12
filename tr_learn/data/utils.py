import pathlib
import warnings

import cv2
import numpy as np

UNKNOWN_LABEL = "unknown"


def find_circles(image: np.ndarray):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale = cv2.medianBlur(grayscale, 5)

    min_distance = min(image.shape[:2]) // 2
    min_radius = min(image.shape[:2]) // 4

    circles = cv2.HoughCircles(grayscale,
                               cv2.HOUGH_GRADIENT,
                               dp=1,
                               minDist=min_distance,
                               minRadius=min_radius)

    if circles is None:
        warnings.warn("Cannot find any circle. Fill with default centered circle", UserWarning)
        circles = np.array(
            [
                [
                    [image.shape[1] // 2, image.shape[0] // 2,
                        round(0.75 * min(image.shape[:2])) // 2]
                ]
            ], dtype=np.int32
        )

    return circles[0].round().astype(np.int32)


def filter_circles(circles: np.ndarray, image_width: int, image_height: int):
    start_x, end_x = round(0.25 * image_width), round(0.75 * image_width)
    start_y, end_y = round(0.25 * image_height), round(0.75 * image_height)

    circle_mask = (circles[:, 0] >= start_x) & (circles[:, 1] <= end_x) & (
        circles[:, 1] >= start_y) & (circles[:, 1] < end_y)

    if circle_mask.any():
        circles = circles[circle_mask]

    circle_most_radius_index = circles[:, 2].argmax()

    return circles[circle_most_radius_index]


def extract_plate(image: np.ndarray):
    circles = find_circles(image)
    circles = filter_circles(circles, image.shape[1], image.shape[0])
    mask = segment_plate(image, circles)
    image = image * mask[..., np.newaxis]
    xy_wh = cv2.boundingRect(mask)
    return image[xy_wh[1]: xy_wh[1] + xy_wh[3], xy_wh[0]: xy_wh[0] + xy_wh[2]]


def segment_plate(image: np.ndarray, circle: np.ndarray):
    mask = np.full(image.shape[:2], cv2.GC_BGD, dtype=np.uint8)

    bound_rect_xy_xy = (circle[0] - circle[2],
                        circle[1] - circle[2],
                        circle[0] + circle[2],
                        circle[1] + circle[2])

    cv2.rectangle(mask,
                  bound_rect_xy_xy[:2],
                  bound_rect_xy_xy[2:],
                  cv2.GC_PR_FGD,
                  cv2.FILLED)

    cv2.circle(mask, circle[:2], round(0.9 * circle[2]), cv2.GC_FGD, cv2.FILLED)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    return cv2.erode(mask, kernel, iterations=5)


def get_split_and_class(path: pathlib.Path):
    if path.parent.name == "test":
        return path.parent.name, UNKNOWN_LABEL

    return path.parent.parent.name, path.parent.name
