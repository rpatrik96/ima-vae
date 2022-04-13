import cv2
import numpy as np


def projective_transform(X: np.ndarray) -> np.ndarray:
    print("Applying projective transformation...")
    rows, cols = X.shape[1], X.shape[2]
    src_points = np.float32(
        [[0, 0], [0, rows - 1], [cols / 2, 0], [cols / 2, rows - 1]]
    )
    dst_points = np.float32(
        [[8, 3], [4, rows - 17], [cols / 2 + 10, 15], [cols / 2 + 8, rows - 26]]
    )
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    X = np.array([cv2.warpPerspective(x, projective_matrix, (cols, rows)) for x in X])

    return X


def affine_transform(X: np.ndarray) -> np.ndarray:
    print("Applying affine transformation...")

    rows, cols = X.shape[1], X.shape[2]
    src_points = np.float32(
        [[rows // 4, cols // 4], [rows, cols // 4], [rows // 4, cols - 16]]
    )
    dst_points = np.float32(
        [[rows // 6, cols // 2], [rows, cols // 4], [rows // 2, cols]]
    )
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)

    X = np.array([cv2.warpAffine(x, affine_matrix, (cols, rows)) for x in X])

    return X


def hsv_change(
    img: np.ndarray, delta_h: int = 0, delta_s: int = 0, delta_v: int = 0
) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    h = cv2.add(h, delta_h)
    s = cv2.add(s, delta_s)
    v = cv2.add(v, delta_v)

    hsv = cv2.merge([h, s, v])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb
