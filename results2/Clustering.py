from Coordinate_trans import load_data
from Coordinate_trans import person_score
from Coordinate_trans import coordinate_transform
from Coordinate_trans import draw_pose
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    load = load_data()
    score = person_score(load)
    after = coordinate_transform(load)
    # draw_pose(after, score)


# 处理聚类的输入
input_list = []
for item in after:
    input = np.hstack((item[0], item[1]))
    input_list.append(input)
input_array = np.array((input_list))
# print(np.shape(input_array))


# count_false_dict = {}
input_with_filter = []
for i in range(len(input_array)):
    count_nan = np.isnan(input_array[i])
    count = count_nan.tolist().count(True)

    if count == 0:
        input_with_filter.append(input_array[i].tolist())
# print(input_with_filter)
input_with_filter_array = np.array(input_with_filter)
# print(np.shape(input_with_filter_array))
# # 数缺各节点的个数有多少
#     if count in count_false_dict.keys():
#         count_false_dict[count] += 1
#     else:
#         count_false_dict.update({count: 1})
#
# print(count_false_dict)

# Kmeans 聚类
k = 3
iteration = 1000
# data = 1.0 * (input_with_filter_array - np.mean(input_with_filter_array)) / np.std(input_with_filter_array)
model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
model.fit(input_with_filter_array)

r1 = pd.Series(model.labels_).value_counts()
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2, r1], axis=1)
# r.columns = list(data.columns)
print(r)

r_array = np.array(r)
# print(r_array)
# 绘制聚类中心
for i in range(len(r_array)):
    plt.figure()
    for j in range(0, 16):
        plt.scatter(r_array[i][j], -r_array[i][j+17], c='r')

    plt.plot([r_array[i][0], r_array[i][1]], [-r_array[i][17], -r_array[i][18]])
    plt.plot([r_array[i][0], r_array[i][2]], [-r_array[i][17], -r_array[i][19]])
    plt.plot([r_array[i][1], r_array[i][3]], [-r_array[i][18], -r_array[i][20]])
    plt.plot([r_array[i][2], r_array[i][4]], [-r_array[i][19], -r_array[i][21]])
    plt.plot([r_array[i][5], r_array[i][6]], [-r_array[i][22], -r_array[i][23]])
    plt.plot([r_array[i][5], r_array[i][7]], [-r_array[i][22], -r_array[i][24]])
    plt.plot([r_array[i][7], r_array[i][9]], [-r_array[i][24], -r_array[i][26]])
    plt.plot([r_array[i][6], r_array[i][8]], [-r_array[i][23], -r_array[i][25]])
    plt.plot([r_array[i][8], r_array[i][10]], [-r_array[i][25], -r_array[i][27]])
    plt.plot([r_array[i][11], r_array[i][12]], [-r_array[i][28], -r_array[i][29]])
    plt.plot([r_array[i][5], r_array[i][11]], [-r_array[i][22], -r_array[i][28]])
    plt.plot([r_array[i][6], r_array[i][12]], [-r_array[i][23], -r_array[i][29]])
    plt.plot([r_array[i][12], r_array[i][14]], [-r_array[i][29], -r_array[i][31]])
    plt.plot([r_array[i][14], r_array[i][16]], [-r_array[i][31], -r_array[i][33]])
    plt.plot([r_array[i][11], r_array[i][13]], [-r_array[i][28], -r_array[i][30]])
    plt.plot([r_array[i][13], r_array[i][15]], [-r_array[i][30], -r_array[i][32]])

plt.show()
