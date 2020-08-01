import torch
from action_detect.data import *
from action_detect.net import *
import cv2

DEVICE = "cpu"


def action_detect(net,pose,crown_proportion):
    # img = cv2.cvtColor(pose.img_pose,cv2.IMREAD_GRAYSCALE)

    img = pose.img_pose.reshape(-1)
    img = img / 255  # 把数据转成[0,1]之间的数据

    img = np.float32(img)

    img = torch.from_numpy(img[None,:]).cpu()

    predect = net(img)

    action_id = int(torch.argmax(predect,dim=1).cpu().detach().item())

    possible_rate = 0.6*predect[:,action_id] + 0.4*(crown_proportion-1)

    possible_rate = possible_rate.detach().numpy()[0]

    if possible_rate > 0.55:
        pose.pose_action = 'fall'
        if possible_rate > 1:
            possible_rate = 1
        pose.action_fall = possible_rate
        pose.action_normal = 1-possible_rate
    else:
        pose.pose_action = 'normal'
        if possible_rate >= 0.5:
            pose.action_fall = 1-possible_rate
            pose.action_normal = possible_rate
        else:
            pose.action_fall = possible_rate
            pose.action_normal = 1 - possible_rate

    # if int(action_id) == 0:
    #     pose.pose_action = 'fall'
    # else:
    #     pose.pose_action = 'normal'

    # print(pose.action_fall, pose.action_normal,pose.pose_action)

    return pose


if __name__ == '__main__':

    net = NetV2()
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
