
import glob
import shutil

root_path = '/bag_data/cam1'

img_path_list = sorted(glob.glob(f"{root_path}/*.jpg"))

for i_m_n in img_path_list:
    img_idx = i_m_n.split('/')[-1].split('.')[0]
    img_idx = int(img_idx)
    if img_idx % 2 == 0:
        shutil.move(i_m_n, f'/home/coolshen/Desktop/code/my_code/paper_implement/bag_data/cam2/{img_idx}.jpg')
