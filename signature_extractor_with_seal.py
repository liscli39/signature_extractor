"""Extract signatures from an image."""

import cv2
import numpy as np
from signature_extractor import extractor


def extractor_with_seal(origin):
    # increase color intensity
    img = cv2.threshold(origin, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # create red mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    im_red_ball_mask_1 = cv2.inRange(hsv, (0, 100, 0), (10, 255, 255))
    im_red_ball_mask_2 = cv2.inRange(hsv, (170, 100, 0), (180, 255, 255))
    im_red_ball_mask_full = im_red_ball_mask_1 + im_red_ball_mask_2

    # mask red color with white
    img = cv2.bitwise_and(origin, origin, mask=255 - im_red_ball_mask_full)
    img[np.where((img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    result = extractor(img)
    return result


