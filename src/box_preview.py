"""Fit a complete preview into the window without cropping or stretching it."""
import cv2


def fit_preview(image, width=1280, height=720):
    scale = min(1.0, width / image.shape[1], height / image.shape[0])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
