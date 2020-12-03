import os
import cv2
from signature_extractor import extractor
from signature_extractor_with_seal import extractor_with_seal

for root, dirs, files in os.walk('./inputs', topdown=False):
    for name in files:
        print(os.path.join(root, name))
        img_input = cv2.imread(os.path.join(root, name))

        # get first result
        result = extractor_with_seal(img_input)

        cv2.imwrite("./outputs/{}".format(name), result)
