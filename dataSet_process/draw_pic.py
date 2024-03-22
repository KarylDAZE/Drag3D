import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']
data = [0.895923138, 0.90273571, 0.901218643, 0.908375492, 0.908762207,
        0.903372269, 0.909729462, 0.900340767, 0.90639183, 0.908523293]

x = range(1, len(data) + 1)  # 创建x轴坐标序列

plt.scatter(x, data, label='差值大小')
plt.axhline(y=data[0], color='r', linestyle='--', label='参考线')

plt.xlabel('索引点')
plt.ylabel('SDF差值')
plt.title('SDF差值表')
plt.legend()

# 设置y轴范围，使得图更明显地呈现几乎平行的效果
plt.ylim(0, 1)

plt.grid(True, axis='y', which='both')  # 添加网格线以便观察变化
plt.show()