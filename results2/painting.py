import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

with open(r'C:\Bravo\Human Pose Estimation\results2\alphapose-results.json', encoding='utf-8') as f:
    j_data = json.load(f)
    print('图片关节点数据：')
    for img in j_data:
        print('=>', img)
    print('-'*1500)

data_joint = [item['keypoints'] for item in j_data]
data_joint_array = np.array(data_joint)
pic_num = len(data_joint_array)
print('关节点信息:')
print(data_joint_array)
print('data_joint_shape: ', np.shape(data_joint_array))
print('-'*1500)


#   绘制坐标轴
# draw.line((o1_x_array[4], o1_y_array[4], o2_x_array[4], o2_y_array[4]), fill=(255, 0, 0), width=3)
# draw.line((data_joint_array[4][15], data_joint_array[4][16], data_joint_array[4][18], data_joint_array[4][19]), fill=(255, 0, 0), width=3)

#   绘制关节点
im_list = []

for item in range(pic_num):
    im = Image.open("C:\\Bravo\\Human Pose Estimation\\photos2\\{}.jpg".format(item + 1))
    im_list.append(im)
    font_en = ImageFont.truetype('C:\\Bravo\\Human Pose Estimation\\results2\\Arial.ttf', 15)
    draw = ImageDraw.Draw(im_list[item])
    for index in range(0, 51, 3):
        draw.ellipse([(data_joint_array[item][index] - 2, data_joint_array[item][index+1] - 2), (data_joint_array[item][index] + 2, data_joint_array[item][index+1] + 2)], fill=(0, 255, 0))
        # draw.point([(data_joint_array[0][index], data_joint_array[0][index+1])], fill=(0, 255, 0))
        draw.text((data_joint_array[item][index], data_joint_array[item][index + 1]), u'{}'.format(index//3), 'red', font_en)
        draw.text((data_joint_array[item][index], data_joint_array[item][index + 1] + 15), u'{:.2f}'.format(data_joint_array[item][index + 2]), 'green', font_en)
    im.show()

del draw

