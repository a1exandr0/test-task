import cv2
import requests
import json
import matplotlib.pyplot as plt
import numpy as np

src_path = "/Insert/Src/Image/Path/Here"
ref_path = "/Insert/REF/Image/Path/Here"
label = 1
src_img = cv2.imread(src_path)[:,:,::-1]
ref_img = cv2.imread(ref_path)[:,:,::-1]
data = {"src":src_img.tolist(), "ref":ref_img.tolist(), "ref_label":label}

resp = requests.post("http://localhost:5000/inference", json=data)
resp = json.loads(resp.text)
plt.imshow(np.array(resp["result"]))
plt.show()
