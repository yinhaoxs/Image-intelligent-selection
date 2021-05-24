# -*- coding: utf-8 -*-
"""
# @Date: 2020-06-09 16:22
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: demo.py
# Copyright @ 2020 yinhao. All rights reserved.
"""

import numpy as np
import torch
import cv2
from .detector_db import detect_faces

### 修改代码
def test(img, model, threshold):
    height, width = img.shape[:2]
    ### 检测人脸
    bounding_boxes, _ = detect_faces(img, model, threshold)
    preds, areas, axis = list(), list(), list()
    for box in bounding_boxes:
        x1, y1, x2, y2 = (box[:4]+0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size = int(max([w, h])*1.3)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        cropped = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)


        x1_exp, x2_exp, y1_exp, y2_exp = int(x1-w*0.2), int(x2+w*0.2), int(y1-w*0.4), int(y2+w*0.2)
        # x1_exp, x2_exp, y1_exp, y2_exp = int(x1-w*0.4), int(x2+w*0.4), int(y1-w*0.8), int(y2+w*0.4)
        x1_exp = max(0, x1_exp)
        x2_exp = min(width, x2_exp)
        y1_exp = max(0, y1_exp)
        y2_exp = min(height, y2_exp)

        cropped_exp = img[y1_exp:y2_exp, x1_exp:x2_exp]
        preds.append(cropped_exp)
        areas.append((y2-y1)*(x2-x1))
        axis.append([x1_exp, y1_exp, x2_exp, y2_exp])

    try:
        max_face_index = np.argmax(areas)
        out1, out2 = preds[max_face_index], axis[max_face_index]
    except:
        out1, out2 = None, None

    return out1, out2


if __name__ == "__main__":
    pass