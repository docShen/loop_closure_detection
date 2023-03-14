import numpy as np

txt_path = '/bag_data/raw_data/NewCollegeTextFormat.txt'

empt_arr_list = []
with open(txt_path, 'r') as f:
    txt = f.readlines()
    for i_idx,i in enumerate(txt):
        if i_idx % 2 == 0:
            continue
        i = i.split(',')
        i = [int(j) for j_idx,j in enumerate(i) if j_idx % 2 != 0 ]
        txt_np = np.array(i)
        empt_arr_list.append(txt_np)
empt_arr = np.array(empt_arr_list).astype(np.int)

for i_idx,i in enumerate(empt_arr):
    for j_idx,j in enumerate(i):
        if i_idx == j_idx:
            empt_arr[i_idx][j_idx] = 1

np.savetxt('/home/coolshen/Desktop/code/my_code/paper_implement/bag_data/empt_arr.txt', empt_arr, fmt='%d')

print(empt_arr_list)