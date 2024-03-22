from typing import List
import json
import numpy as np

with open('../data_backup/sdf_output.txt', 'r') as f:
    arrays = []
    for line in f.readlines():
        data = json.loads(line)
        arr = np.array(data)
        arr = arr.squeeze()
        # print(arr.shape)
        arrays.append(arr)
    print(arrays)
    print(len(arrays))
    max_arr = []
    for i in range(1, len(arrays)):
        arr = arrays[i] - arrays[i - 1]
        # 计算数组的绝对值
        abs_arr = np.abs(arr)

        # 找到绝对值最大的元素的索引
        max_abs_index = np.argmax(abs_arr)
        print(f"绝对值最大值为 {arr[max_abs_index]}，其索引为 {max_abs_index}")
        max_arr.append(abs_arr[max_abs_index])
    max_arr.sort()
    print(max_arr)