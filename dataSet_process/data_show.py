import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']

# 数据
data = {
    '车辆(0.4)': {'Cov-CD(%)': [58.39, 58.7], 'MMD-CD': [0.72, 0.7], 'FID': [10.26, 10.22]},
    '车辆(0.6)': {'Cov-CD(%)': [52.29, 57.31], 'MMD-CD': [1.43, 1.25], 'FID': [20.09, 23.28]},
    '车辆(0.8)': {'Cov-CD(%)': [43.13, 50.35], 'MMD-CD': [4.43, 3.12], 'FID': [35.7, 26.59]},
    '椅子(0.3)': {'Cov-CD(%)': [69.91, 70.02], 'MMD-CD': [3.72, 3.65], 'FID': [23.28, 23.14]},
    '椅子(0.45)': {'Cov-CD(%)': [62.48, 68.77], 'MMD-CD': [6.71, 5.89], 'FID': [27.58, 25.33]},
    '椅子(0.6)': {'Cov-CD(%)': [54.8, 51.76], 'MMD-CD': [8.12, 8.53], 'FID': [31.48, 32.65]},
}

# 定义类别和指标
categories = list(data.keys())
metrics = ['Cov-CD(%)', 'MMD-CD', 'FID']

# 创建图形
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), sharey=True)

for ax, metric in zip(axs.flat, metrics):
    for i, category in enumerate(categories):
        values = data[category][metric]
        ax.bar(i + 0.15, values[0], width=0., label='本系统')
        ax.bar(i + 0.35, values[1], width=0., label='Drag3D')

    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_title(metric)
    ax.legend()

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', xytext=(0, 0), textcoords='offset points')

fig.tight_layout()
plt.show()