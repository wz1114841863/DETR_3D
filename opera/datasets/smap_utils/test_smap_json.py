# Document function description
#   用来查看json文件中具体的数据格式。

import json
import cv2 as cv


def load_smap_coco():
    smap_coco_path = "/data/coco/annotations/coco_keypoints_train2017.json"
    with open(file=smap_coco_path, mode='r') as fp:
        annos = json.load(fp)
    # print(len(annos))
    # print(len(annos['root']))
    return annos

def load_smao_muco():
    smap_muco_path = "/data/MuCo/annotations/MuCo.json"
    with open(file=smap_muco_path, mode='r') as fp:
        annos = json.load(fp)
    # print(len(annos['root']))
    return annos

if __name__ == "__main__":
    annos = load_smap_coco()
    annos = annos['root']
    img_prefix = "/data/coco/"
    img_path = img_prefix + annos[0]['img_paths']
    print(f"img_path is {img_path}")
    bboxs = annos[0]['bboxs']
    print(f"the number of person in img is {len(bboxs)}")
    # 绘制矩形框
    img = cv.imread(img_path, 1)
    print(f"img shape: {img.shape}")
    for bbox in bboxs:
        print(bbox)
        ptLeftTop = (int(bbox[0]), int(bbox[1]))
        ptRightBottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        img = cv.rectangle(img, ptLeftTop, ptRightBottom, (0, 0, 255))
    