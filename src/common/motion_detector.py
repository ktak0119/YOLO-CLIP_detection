"""Pluggable per-frame motion-detection backend for Stage1.

Ported from 2026624_ClaudeDetection/work/motion_detector.py. Stage1's role in
this pipeline is narrower than that project's: it only flags whether a frame
has >=1 qualifying candidate (min_area <= area <= max_area) -- it never picks
a "best" contour and never writes a crop. Semantic/score-based selection
happens downstream in Stage2 (YOLO) and Stage3 (CLIP).

MOG2Detector must be called on every frame of a video, in order, without
being reset at bin boundaries (it's a streaming background model).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidate:
    area: float
    bbox: tuple  # (x, y, w, h) in the coordinate space of the frame passed to detect()
    contour: object


class MotionDetector:
    """Interface. Call detect() once per frame, in order, for the whole video."""

    def detect(self, gray_frame):
        raise NotImplementedError


class MOG2Detector(MotionDetector):
    """Stage1 backend.

    Known recall risks (not solved here, must be validated by the user later
    via a recall harness): (1) a target that stops moving for a while can be
    absorbed into the background model over time; (2) the model is unstable
    for the first few frames after construction ("warm-up"). `warmup_frames`
    marks those frames as background-learning-only (no candidates emitted).
    """

    def __init__(self, history=300, var_threshold=24, learning_rate=-1,
                 min_area=100, max_area=25000, mask_top_frac=0.0, warmup_frames=0):
        self.backsub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self.learning_rate = learning_rate
        self.min_area = min_area
        self.max_area = max_area
        self.mask_top_frac = mask_top_frac
        self.warmup_frames = warmup_frames
        self._frame_count = 0
        self._open_kernel = np.ones((3, 3), np.uint8)
        self._dilate_kernel = np.ones((3, 3), np.uint8)

    def detect(self, gray_frame):
        self._frame_count += 1
        fg = self.backsub.apply(gray_frame, learningRate=self.learning_rate)
        if self.mask_top_frac > 0:
            mask_rows = int(gray_frame.shape[0] * self.mask_top_frac)
            if mask_rows > 0:
                fg[:mask_rows, :] = 0
        if self._frame_count <= self.warmup_frames:
            # Background model is still being learned; suppress candidates
            # rather than emit noise, but keep feeding frames to backsub.apply()
            # above so learning isn't interrupted.
            return []
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._open_kernel)
        fg = cv2.dilate(fg, self._dilate_kernel, iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                out.append(Candidate(area=area, bbox=cv2.boundingRect(c), contour=c))
        return out
