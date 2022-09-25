import os
import os.path as osp
import numpy as np
import json


root_dir = '/data/MuCo/'
anno_name = 'MuCo-3DHP.json'
anno_file = os.path.join(root_dir, 'annotations', anno_name)

output_path = "/data/jupyter/DEKR_3D/preprocess/"
output_json_file = os.path.join(output_path, 'MuCo.json')

muco2mpi15 = [1, 0, 14, 5, 6, 7, 11, 12, 13, 2, 3, 4, 8, 9, 10]

def main():
    # if not osp.exists(output_json_file):
        # os.system("mkdir -p %s" % (output_json_file))
    with open(anno_file, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    annot_dict = dict()
    for i, annot in enumerate(annotations):
        img_id = annot['image_id']
        if img_id not in annot_dict.keys():
            annot_dict[img_id] = dict()
            annot_dict[img_id]['bboxs'] = []
            annot_dict[img_id]['bodys'] = []
            annot_dict[img_id]['areas'] = []
            
        p2d = np.array(annot['keypoints_img'])[muco2mpi15, :]
        p3d = np.array(annot['keypoints_cam'])[muco2mpi15, :]
        vis = np.array(annot['keypoints_vis'])[muco2mpi15]
        num_vis = vis.sum()
        if num_vis == 0:
            # print(f"img_id:{img_id}, num_vis: {num_vis}")
            continue
        pose = []
        for j in range(15):
            pose_info = [p2d[j, 0], p2d[j, 1], p3d[j, 2] / 10, int(vis[j] * 2), p3d[j, 0] / 10, p3d[j, 1] / 10,
                            p3d[j, 2] / 10]
            pose.append(pose_info)
        annot_dict[img_id]['bodys'].append(pose)
        annot_dict[img_id]['bboxs'].append(annot['bbox'])
        area = annot['bbox'][2] * annot['bbox'][3]  # 用bbox替代面积
        annot_dict[img_id]['areas'].append(area)

    output_json = dict()
    output_json['root'] = []
    count = 0
    for img_info in data['images']:
        count += 1
        cur_info = dict()
        cur_info['img_height'] = img_info['height']
        cur_info['img_width'] = img_info['width']
        cur_info['img_paths'] = osp.join('images', img_info['file_name'])
        cur_info['dataset'] = "MuCo"
        cur_info['isValidation'] = 0
        fx, fy = img_info['f']
        cx, cy = img_info['c']
        cur_info['bodys'] = annot_dict[img_info['id']]['bodys']
        cur_info['bboxs'] = annot_dict[img_info['id']]['bboxs']
        cur_info['areas'] = annot_dict[img_info['id']]['areas']
        for nh in range(len(cur_info['bodys'])):
            for j in range(len(cur_info['bodys'][nh])):
                cur_info['bodys'][nh][j] += [fx, fy, cx, cy]
        output_json['root'].append(cur_info)
    print(f"处理了{count}张图片")
    with open(output_json_file, 'w') as f:
        json.dump(output_json, f)
    print("finished.")
    
if __name__ == "__main__":
    main()