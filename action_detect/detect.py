import torch
from action_detect.data import *
from action_detect.net import *
import cv2

DEVICE = "cpu"


def action_detect(net,img):
    # img = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)

    img = img.reshape(-1)
    img = img / 255  # 把数据转成[0,1]之间的数据

    img = np.float32(img)

    img = torch.from_numpy(img[None,:]).cpu()

    predect = net(img)

    return torch.argmax(predect,dim=1).cpu().detach().item()


if __name__ == '__main__':

    net = NetV3()
    # 加载已训练的数据
    net.load_state_dict(torch.load("D:/py/openpose_lightweight/action_detect/checkPoint/action.pt"))
    net.to(DEVICE)  # 使用GPU进行训练

    img = cv2.imread('C:/Users/lieweiai/Desktop/human_pose/test/normal/1595825240980.jpg', cv2.IMREAD_GRAYSCALE)  # 以灰度图形式读数据
    # img = img.reshape(-1)
    # img = img / 255  # 把数据转成[0,1]之间的数据
    #
    # img = np.float32(img)
    #
    # img = torch.from_numpy(img[None,:]).cuda()
    #
    # predect = net(img)

    print(action_detect(net,img))
