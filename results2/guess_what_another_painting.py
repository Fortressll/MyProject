import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import Coordinate_trans

with open(r'C:\Bravo\Human Pose Estimation\results2\test_alphapose-results_new.json', encoding='utf-8') as f:
    j_data = json.load(f)

data_joint = [item['keypoints'] for item in j_data]
data_joint_array = np.array(data_joint)

im = Image.open(r"C:\Bravo\Human Pose Estimation\results2\test.jpg")
for item in range(0, 10):
    font_en = ImageFont.truetype('C:\\Bravo\\Human Pose Estimation\\results2\\Arial.ttf', 15)
    draw = ImageDraw.Draw(im)
    for index in range(0, 51, 3):
        draw.ellipse([(data_joint_array[item][index] - 2, data_joint_array[item][index+1] - 2), (data_joint_array[item][index] + 2, data_joint_array[item][index+1] + 2)], fill=(255, 0, 0))
        draw.line((data_joint_array[item][0], data_joint_array[item][1], data_joint_array[item][3], data_joint_array[item][4]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][0], data_joint_array[item][1], data_joint_array[item][6], data_joint_array[item][7]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][3], data_joint_array[item][4], data_joint_array[item][9], data_joint_array[item][10]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][6], data_joint_array[item][7], data_joint_array[item][12], data_joint_array[item][13]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][15], data_joint_array[item][16], data_joint_array[item][18], data_joint_array[item][19]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][15], data_joint_array[item][16], data_joint_array[item][21], data_joint_array[item][22]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][21], data_joint_array[item][22], data_joint_array[item][27], data_joint_array[item][28]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][18], data_joint_array[item][19], data_joint_array[item][24], data_joint_array[item][25]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][24], data_joint_array[item][25], data_joint_array[item][30], data_joint_array[item][31]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][33], data_joint_array[item][34], data_joint_array[item][36], data_joint_array[item][37]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][33], data_joint_array[item][34], data_joint_array[item][39], data_joint_array[item][40]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][39], data_joint_array[item][40], data_joint_array[item][45], data_joint_array[item][46]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][36], data_joint_array[item][37], data_joint_array[item][42], data_joint_array[item][43]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][42], data_joint_array[item][43], data_joint_array[item][48], data_joint_array[item][49]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][15], data_joint_array[item][16], data_joint_array[item][33], data_joint_array[item][34]), fill=(0, 255, 0))
        draw.line((data_joint_array[item][18], data_joint_array[item][19], data_joint_array[item][36], data_joint_array[item][37]), fill=(0, 255, 0))
        # draw.point([(data_joint_array[0][index], data_joint_array[0][index+1])], fill=(0, 255, 0))
        # draw.text((data_joint_array[item][index], data_joint_array[item][index + 1]), u'{}'.format(index//3), 'red', font_en)
        # draw.text((data_joint_array[item][index], data_joint_array[item][index + 1] + 15), u'{:.2f}'.format(data_joint_array[item][index + 2]), 'green', font_en)
im.show()

del draw

