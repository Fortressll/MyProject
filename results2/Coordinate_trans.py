import json
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    with open(r'C:\Bravo\Human Pose Estimation\results2\test.json', encoding='utf-8') as f:
        j_data = json.load(f)
        # print('图片数据：')
        # for img in j_data:
        #     print('=>', img)
        # print('-'*1500)

        return j_data


def coordinate_transform(j_data):
    joint_threshold = 0.01
    data_joint = [item['keypoints'] for item in j_data]
    data_joint_array = np.array(data_joint)
    # print('关节点信息:')
    # print(data_joint_array)
    # print('data_joint_shape: ', np.shape(data_joint_array))
    # print('-' * 1500)


    # print('最差关节点' * 6)
    # for index in range(2, 51, 3):
    #     print(data_joint_array[13485][index])
    # print('最差关节点' * 6)

    # #求每个人所有关节点置信度的和，看舍弃
    # joint_score_list = []
    # rubbish = []
    # for item in data_joint_array:
    #     joint_score = 0
    #     for index in range(2, 51, 3):
    #         joint_score += item[index]
    #     joint_score_list.append(joint_score)
    # print('关节点舍弃：')
    # for item in joint_score_list:
    #     if item < 0.85:
    #         rubbish.append(item)
    # print(np.shape(rubbish))


    # 把置信度过低的关节点进行标记
    for item in data_joint_array:
        for index in range(2, 51, 3):
            if item[index] < joint_threshold:
                item[index - 2] = None
                item[index - 1] = None

    # 计算两肩关节和两髋关节各自的中点
    pic_num = len(data_joint_array)
    x1_sum_list = []
    x2_sum_list = []
    y1_sum_list = []
    y2_sum_list = []

    for item in data_joint_array:
        x1_sum_list.append(item[15] + item[18])
        x2_sum_list.append(item[33] + item[36])
        y1_sum_list.append(item[16] + item[19])
        y2_sum_list.append(item[34] + item[37])

    x1_sum_array = np.array(x1_sum_list)
    x2_sum_array = np.array(x2_sum_list)
    y1_sum_array = np.array(y1_sum_list)
    y2_sum_array = np.array(y2_sum_list)

    o1_x_array = x1_sum_array / 2.0
    o1_y_array = y1_sum_array / 2.0
    o2_x_array = x2_sum_array / 2.0
    o2_y_array = y2_sum_array / 2.0

    re_o2_x_array = o2_x_array - o1_x_array
    re_o2_y_array = o2_y_array - o1_y_array

    # print('肩关节中心点为：')
    # for i in range(len(j_data)):
    #     print('({}, {})'.format(o1_x_array[i], o1_y_array[i]))
    # print('髋关节中心点为：')
    # for i in range(len(j_data)):
    #     print('({}, {})'.format(o2_x_array[i], o2_y_array[i]))
    # print('-'*1500)


    # 将原坐标系下的各关节点坐标投影在以肩关节中心点为原点的直角坐标系上
    re_data_joint_x_list = []
    re_data_joint_y_list = []

    for item in range(pic_num):
        for index in range(0, 51, 3):
            re_data_joint_x = data_joint_array[item][index] - o1_x_array[item]
            re_data_joint_y = o1_y_array[item] - data_joint_array[item][index + 1]
            re_data_joint_x_list.append(re_data_joint_x)
            re_data_joint_y_list.append(re_data_joint_y)
    re_data_joint_x_array = np.array(re_data_joint_x_list).reshape(pic_num, 17)
    re_data_joint_y_array = np.array(re_data_joint_y_list).reshape(pic_num, 17)

    # print(np.shape(re_data_joint_x_array))
    # print('x:', re_data_joint_x_array)
    # print('Y:', re_data_joint_y_array)


    # 处理输入
    before_list = []

    for item in range(pic_num):
        before = np.vstack((re_data_joint_x_array[item], re_data_joint_y_array[item]))
        before_list.append(before)

    before_array = np.array(before_list)
    # print(np.shape(before_array))
    # print(before_array)

    # 计算基
    matrix_trans_list = []
    matrix_trans_reverse = []

    for item in range(pic_num):
        m1 = np.vstack((re_data_joint_x_array[item][5], re_data_joint_y_array[item][5]))
        m2 = np.vstack((- re_o2_x_array[item], - re_o2_y_array[item]))
        matrix_trans = np.hstack((m1, m2))
        matrix_trans_list.append(matrix_trans)

    matrix_trans_array = np.array(matrix_trans_list)

    for i in range(pic_num):
        # 过滤掉奇异矩阵
        if np.linalg.det(matrix_trans_array[i]) != 0:
            matrix_trans_each = np.linalg.inv(matrix_trans_array[i])
            # print(matrix_trans_each)
            matrix_trans_reverse.append(matrix_trans_each)

    matrix_trans_array_reverse = np.array(matrix_trans_reverse)

    # print(matrix_trans_array_reverse)
    # print(np.shape(matrix_trans_array_reverse))


    # 转换坐标
    new_pic_num = len(matrix_trans_array_reverse)
    after_list = []

    for item in range(new_pic_num):
        after = np.dot(matrix_trans_array_reverse[item], before_array[item])
        after_list.append(after)
    after_array = np.array(after_list)
    # print('转换后的坐标维度为：', np.shape(after_array))
    # # print(after_array)

    return after_array

def draw_pose(after_array, sort_score):

    for item in range(len(after_array)):
        plt.figure()
        for i in range(0, 17):
            plt.scatter(after_array[item][0][i], -after_array[item][1][i], c='r')

        plt.plot([after_array[item][0][0], after_array[item][0][1]], [-after_array[item][1][0], -after_array[item][1][1]])
        plt.plot([after_array[item][0][0], after_array[item][0][2]], [-after_array[item][1][0], -after_array[item][1][2]])
        plt.plot([after_array[item][0][1], after_array[item][0][3]], [-after_array[item][1][1], -after_array[item][1][3]])
        plt.plot([after_array[item][0][2], after_array[item][0][4]], [-after_array[item][1][2], -after_array[item][1][4]])
        plt.plot([after_array[item][0][5], after_array[item][0][6]], [-after_array[item][1][5], -after_array[item][1][6]])
        plt.plot([after_array[item][0][5], after_array[item][0][7]], [-after_array[item][1][5], -after_array[item][1][7]])
        plt.plot([after_array[item][0][7], after_array[item][0][9]], [-after_array[item][1][7], -after_array[item][1][9]])
        plt.plot([after_array[item][0][6], after_array[item][0][8]], [-after_array[item][1][6], -after_array[item][1][8]])
        plt.plot([after_array[item][0][8], after_array[item][0][10]], [-after_array[item][1][8], -after_array[item][1][10]])
        plt.plot([after_array[item][0][5], after_array[item][0][11]], [-after_array[item][1][5], -after_array[item][1][11]])
        plt.plot([after_array[item][0][6], after_array[item][0][12]], [-after_array[item][1][6], -after_array[item][1][12]])
        plt.plot([after_array[item][0][11], after_array[item][0][12]], [-after_array[item][1][11], -after_array[item][1][12]])
        plt.plot([after_array[item][0][11], after_array[item][0][13]], [-after_array[item][1][11], -after_array[item][1][13]])
        plt.plot([after_array[item][0][13], after_array[item][0][15]], [-after_array[item][1][13], -after_array[item][1][15]])
        plt.plot([after_array[item][0][12], after_array[item][0][14]], [-after_array[item][1][12], -after_array[item][1][14]])
        plt.plot([after_array[item][0][14], after_array[item][0][16]], [-after_array[item][1][14], -after_array[item][1][16]])
    ####################################################
    # count = 0
    # for item in sort_score:
    #     if count > 5:
    #         break
    #     plt.figure()
    #     for i in range(0, 17):
    #         plt.scatter(after_array[item[0]][0][i], -after_array[item[0]][1][i], c='r')
    #     count += 1
    #
    #     plt.plot([after_array[item[0]][0][0], after_array[item[0]][0][1]],[-after_array[item[0]][1][0], -after_array[item[0]][1][1]])
    #     plt.plot([after_array[item[0]][0][0], after_array[item[0]][0][2]], [-after_array[item[0]][1][0], -after_array[item[0]][1][2]])
    #     plt.plot([after_array[item[0]][0][1], after_array[item[0]][0][3]], [-after_array[item[0]][1][1], -after_array[item[0]][1][3]])
    #     plt.plot([after_array[item[0]][0][2], after_array[item[0]][0][4]], [-after_array[item[0]][1][2], -after_array[item[0]][1][4]])
    #     plt.plot([after_array[item[0]][0][5], after_array[item[0]][0][6]], [-after_array[item[0]][1][5], -after_array[item[0]][1][6]])
    #     plt.plot([after_array[item[0]][0][5], after_array[item[0]][0][7]], [-after_array[item[0]][1][5], -after_array[item[0]][1][7]])
    #     plt.plot([after_array[item[0]][0][7], after_array[item[0]][0][9]], [-after_array[item[0]][1][7], -after_array[item[0]][1][9]])
    #     plt.plot([after_array[item[0]][0][6], after_array[item[0]][0][8]], [-after_array[item[0]][1][6], -after_array[item[0]][1][8]])
    #     plt.plot([after_array[item[0]][0][8], after_array[item[0]][0][10]], [-after_array[item[0]][1][8], -after_array[item[0]][1][10]])
    #     plt.plot([after_array[item[0]][0][5], after_array[item[0]][0][11]], [-after_array[item[0]][1][5], -after_array[item[0]][1][11]])
    #     plt.plot([after_array[item[0]][0][6], after_array[item[0]][0][12]], [-after_array[item[0]][1][6], -after_array[item[0]][1][12]])
    #     plt.plot([after_array[item[0]][0][11], after_array[item[0]][0][12]], [-after_array[item[0]][1][11], -after_array[item[0]][1][12]])
    #     plt.plot([after_array[item[0]][0][11], after_array[item[0]][0][13]], [-after_array[item[0]][1][11], -after_array[item[0]][1][13]])
    #     plt.plot([after_array[item[0]][0][13], after_array[item[0]][0][15]], [-after_array[item[0]][1][13], -after_array[item[0]][1][15]])
    #     plt.plot([after_array[item[0]][0][12], after_array[item[0]][0][14]], [-after_array[item[0]][1][12], -after_array[item[0]][1][14]])
    #     plt.plot([after_array[item[0]][0][14], after_array[item[0]][0][16]], [-after_array[item[0]][1][14], -after_array[item[0]][1][16]])


    plt.show()

def person_score(j_data):
    score_list = []

    for img in j_data:
        score = img['score']
        score_list.append(score)

    score_dict = {i + 1: score_list[i]  for i in range(len(score_list))}
    # print(score_dict)

    sort_score = sorted(score_dict.items(), key=lambda item: item[1])
    # print(sort_score)

    return sort_score

# def person_joint_score_sum(data_joint_array):
#     joint_score_sum_list = []
#
#     for img in data_joint_array:
#


