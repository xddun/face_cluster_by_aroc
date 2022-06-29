import json
import os
import traceback

import cv2
import numpy as np
import requests


def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


aimpath = r"/data/dong_xie/facedata/data"
files = all_path(aimpath)

fecedatas = {}
for file in files:
    try:
        img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)  # img是矩阵
        success, encoded_image = cv2.imencode(".jpg", img)
        # 将数组转为bytes
        fb = encoded_image.tobytes()

        res = requests.post(url="http://0.0.0.0:49153/GetFaceVector", files={"file": fb})

        if len(res.json()) == 1:
            fecedatas[os.path.basename(file).split(".")[0]] = res.json()[0][0]
            path6 = os.path.join("/data/dong_xie/face_cluster_by_aroc/data2", file[29:].split("/")[0])
            if not os.path.exists(path6):
                os.mkdir(path6)
            cv2.imwrite(os.path.join("/data/dong_xie/face_cluster_by_aroc/data2", file[29:]), img)
            pass
    except:
        traceback.print_exc()

with open('feature_json/face_features_json.json', 'w', encoding="utf-8") as f:
    f.write(json.dumps(fecedatas, ensure_ascii=False, indent=1))

pass
