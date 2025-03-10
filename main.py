"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/HighPassSkinSmoothing-Python

Title: main.py
Description: A python3 script for high pass skin smoothing
"""


import argparse
import os
import math
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


printc = lambda *_x, _color_code='\033[93m': print(_color_code  + str(_x) + '\033[0m')
pausec = lambda *_x, _color_code='\033[93m': input(_color_code  + str(_x) + '\033[0m')


class HighPassSkinSmoothing(object):
    ratio_smooth = 2.5

    @classmethod
    def get_params(cls, _ndarr_img_bgr: np.ndarray) -> Tuple[float, float]:
        ndarr_img_y = cv2.cvtColor(_ndarr_img_bgr, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
        ndarr_mask_skin = cls.__get_segmented_skin(_ndarr_img_bgr=_ndarr_img_bgr)
        ndarr_skin_y = ndarr_img_y[ndarr_mask_skin == 255]
        val_min, val_max, val_mean, val_std = float(np.min(ndarr_skin_y)), float(np.max(ndarr_skin_y)), float(np.mean(ndarr_skin_y)), float(np.std(ndarr_skin_y))
        level_smooth, level_whiten = val_mean + (cls.ratio_smooth * val_std), val_mean / val_min
        return level_smooth, level_whiten

    @classmethod
    def beautify(cls, _ndarr_img_bgr: np.ndarray, _level_smooth: float = 0.0, _level_whiten: float = 1.0) -> np.ndarray:
        return cls.whiten_skin(
            _ndarr_img_bgr=cls.smoothen_skin(_ndarr_img_bgr=_ndarr_img_bgr, _level_smooth=_level_smooth),
            _level_whiten=_level_whiten
        )

    @classmethod
    def smoothen_skin(cls, _ndarr_img_bgr: np.ndarray, _level_smooth: float = 0.0) -> np.ndarray:
        cls.__is_valid(_ndarr_img_bgr=_ndarr_img_bgr)
        return cls.__smoothen_skin(_ndarr_img_bgr=_ndarr_img_bgr, _level_smooth=_level_smooth)

    @classmethod
    def whiten_skin(cls, _ndarr_img_bgr: np.ndarray, _level_whiten: float = 1.0) -> np.ndarray:
        cls.__is_valid(_ndarr_img_bgr=_ndarr_img_bgr)
        if (1.0 < _level_whiten):
            res = cls.__whiten_skin(_ndarr_img_bgr=_ndarr_img_bgr, _level_whiten=_level_whiten)
        else:
            printc('[{}.whiten_skin] The _level_whiten should be greater than 1.0, not {}.'.format(cls.__name__, _level_whiten), '\033[91m')
            res = _ndarr_img_bgr.copy()
        return res

    @classmethod
    def __whiten_skin(cls, _ndarr_img_bgr: np.ndarray, _level_whiten: float = 1.0) -> np.ndarray:
        level_log_whiten = math.log(_level_whiten)
        printc('[{}.__whiten_skin] level_log_whiten: {}'.format(cls.__name__, level_log_whiten))
        res = np.clip(255.0 * (np.log((_ndarr_img_bgr / 255.0) * (_level_whiten - 1.0) + 1.0) / level_log_whiten), 0.0, 255.0).astype(np.uint8)
        return res

    @classmethod
    def __smoothen_skin(cls, _ndarr_img_bgr: np.ndarray, _level_smooth: float = 0.0) -> np.ndarray:
        printc('[{}.__smoothen_skin] _level_smooth: {}'.format(cls.__name__, _level_smooth))

        img_height, img_width, _ = _ndarr_img_bgr.shape
        ndarr_mask_skin = cls.__get_segmented_skin(_ndarr_img_bgr=_ndarr_img_bgr)
        ndarr_integral, ndarr_integral_sqr = cls.__get_integral(_ndarr_img_bgr=_ndarr_img_bgr)

        ndarr_res_yuv = cv2.cvtColor(_ndarr_img_bgr, cv2.COLOR_BGR2YUV)
        ndarr_tmp_y = ndarr_res_yuv[:, :, 0].astype(np.float32)
        radius = 0.02 * max(img_height, img_width)
        coords = np.column_stack(np.where(ndarr_mask_skin == 255))

        idy_min = np.clip(coords[:, 0] - radius, 1, img_height - 1).astype(np.int64)
        idy_max = np.clip(coords[:, 0] + radius, 1, img_height - 1).astype(np.int64)
        idx_min = np.clip(coords[:, 1] - radius, 1, img_width - 1).astype(np.int64)
        idx_max = np.clip(coords[:, 1] + radius, 1, img_width - 1).astype(np.int64)

        area = (idy_max - idy_min + 1) * (idx_max - idx_min + 1)
        sum_region = ndarr_integral[idy_max, idx_max] - ndarr_integral[idy_min - 1, idx_max] - ndarr_integral[idy_max, idx_min - 1] + ndarr_integral[idy_min - 1, idx_min - 1]
        sum_region_sqr = ndarr_integral_sqr[idy_max, idx_max] - ndarr_integral_sqr[idy_min - 1, idx_max] - ndarr_integral_sqr[idy_max, idx_min - 1] + ndarr_integral_sqr[idy_min - 1, idx_min - 1]
        mean = sum_region / area
        variance = (sum_region_sqr / area) - (mean ** 2)
        k = variance / (variance + _level_smooth)        
        ndarr_tmp_y[coords[:, 0], coords[:, 1]] = np.ceil((k * ndarr_tmp_y[coords[:, 0], coords[:, 1]]) + mean - (k * mean))

        ndarr_res_yuv[:, :, 0] = np.clip(ndarr_tmp_y, 0.0, 255.0).astype(np.uint8)
        ndarr_res_bgr = cv2.cvtColor(ndarr_res_yuv, cv2.COLOR_YUV2BGR)
        return ndarr_res_bgr

    @staticmethod
    def __get_segmented_skin(_ndarr_img_bgr: np.ndarray) -> np.ndarray:
        ndarr_b, ndarr_g, ndarr_r = cv2.split(_ndarr_img_bgr)
        condition_1 = (ndarr_b > 95) & (ndarr_g > 40) & (ndarr_r > 20) & (ndarr_b - ndarr_r > 15) & (ndarr_b - ndarr_g > 15)
        condition_2 = (ndarr_b > 200) & (ndarr_g > 210) & (ndarr_r > 170) & (np.fabs(ndarr_b - ndarr_r) <= 15.0) & (ndarr_b > ndarr_r) & (ndarr_g > ndarr_r)
        conditions = condition_1 | condition_2
        ndarr_mask_skin = np.where(conditions, 255.0, 0.0).astype(np.uint8)
        return ndarr_mask_skin

    @staticmethod
    def __get_integral(_ndarr_img_bgr: np.ndarray) -> tuple:
        ndarr_img_yuv = cv2.cvtColor(_ndarr_img_bgr, cv2.COLOR_BGR2YUV)
        ndarr_img_y = ndarr_img_yuv[:, :, 0].astype(np.uint64)
        integral = np.cumsum(np.cumsum(ndarr_img_y, axis=0), axis=1)
        integral_sqr = np.cumsum(np.cumsum(ndarr_img_y ** 2, axis=0), axis=1)
        return integral, integral_sqr

    @staticmethod
    def __is_valid(_ndarr_img_bgr: np.ndarray) -> None:
        if _ndarr_img_bgr.ndim != 3:
            raise NotImplementedError('[__init__] The _ndarr_img_bgr.ndim should be 3, not {}.'.format(_ndarr_img_bgr.ndim))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img_src', type=str, required=True)
    parser.add_argument('--path_img_dst', type=str, required=True)
    parser.add_argument('--level_auto', action='store_true')
    parser.add_argument('--level_smooth', type=float, default=0.0)
    parser.add_argument('--level_whiten', type=float, default=1.0)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_args()

    path_img_src = Path(args.path_img_src)
    path_img_dst = Path(args.path_img_dst)
    ndarr_img_bgr = cv2.imread(str(path_img_src), cv2.IMREAD_COLOR)

    if isinstance(args.level_auto, (bool, )) and (args.level_auto is True):
        level_smooth, level_whiten = HighPassSkinSmoothing.get_params(_ndarr_img_bgr=ndarr_img_bgr)
    else:
        level_smooth, level_whiten = args.level_smooth, args.level_whiten

    if (isinstance(level_smooth, (int, float, )) is False) or (isinstance(level_whiten, (int, float, )) is False):
        raise ValueError('The both level smooth and level_whiten should be number, not {}, {}.'.format(level_smooth, level_whiten))

    ndarr_skin_refined = HighPassSkinSmoothing.beautify(_ndarr_img_bgr=ndarr_img_bgr, _level_smooth=level_smooth, _level_whiten=level_whiten)
    cv2.imwrite(str(path_img_dst), ndarr_skin_refined)
