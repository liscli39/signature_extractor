"""Extract signatures from an image."""

import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops


def extractor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if region.area >= 250:
            if region.area > the_biggest_component:
                the_biggest_component = region.area

    average = (total_area/counter)
    # a4_constant is used as a threshold value to remove connected pixels
    # are smaller than a4_constant for A4 size scanned documents
    a4_constant = ((average / 84.0) * 250.0) + 100

    # remove the connected pixels are smaller than del_constant
    b = morphology.remove_small_objects(blobs_labels, a4_constant)
    # save the the pre-version which is the image is labelled with colors
    # as considering connected components
    plt.imsave('pre_version.png', b)

    # read the pre-version
    img = cv2.imread('pre_version.png', 0)
    # ensure binary
    result = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)[1]

    return result
