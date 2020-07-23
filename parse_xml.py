from xml.dom.minidom import parse
import os
from PIL import Image,ImageDraw

#此文件为解析精灵标注的xml文件，可以解析相同文件结构的其他xml文件，用来生成训练自己的数据集

path = r"C:\Users\lieweiai\Desktop\action\outputs"

for i in os.listdir(path):

    xml_path = os.path.join(path, i)


# xml_path = r"./data/outputs/6.xml"

    dom = parse(xml_path)
    root = dom.documentElement
    img_name = root.getElementsByTagName("path")[0].childNodes[0].data
    try:
        img_size = root.getElementsByTagName("size")[0]
    except IndexError as e:
        continue

    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_name = img_name.split("\\")
    img_name = img_name[-1].split('.')
    img_name = img_name[0]
    img_size= root.getElementsByTagName("size")[0]
    items = root.getElementsByTagName("item")

    f = open(f"C:/Users/lieweiai/Desktop/action/labels/{img_name}.txt", "a")

    for item in items:
        cls_name = item.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(item.getElementsByTagName("xmin")[0].childNodes[0].data)
        y1 = int(item.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(item.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(item.getElementsByTagName("ymax")[0].childNodes[0].data)
        # w,h = (x2-x1),(y2-y1)
        # cx,cy = x1+int(w/2),y1+int(h/2)
        # print(cls_name, x1, y1, x2, y2)

        if cls_name == 'fall':
            f.write("0 ")
        else:
            f.write("1 ")

        dw = 1. / (int(img_w))
        dh = 1. / (int(img_h))
        x = (x1+ x2) / 2.0 - 1
        y = (y1 + y2) / 2.0 - 1
        w = x2 - x1
        h = y2 - y1
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh

        f.write(f"{x} {y} {w} {h}\n")



        # draw = ImageDraw.Draw(_img_data)
        #
        # draw.rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), outline="red", width=2)
        #
        # _img_data.show()

        # f.flush()
    f.flush()
    f.close()









