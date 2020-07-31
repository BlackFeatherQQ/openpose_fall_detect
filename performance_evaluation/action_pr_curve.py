import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from action_detect.detect import action_detect
from action_detect.data import *
from action_detect.net import *
from torch.utils.data import DataLoader
import torch

DEVICE = "cuda:0"

def plot_action_AP(resultFile):

    labels = []

    pred_possible = []

    pred_class = []

    labels_class = []

    with open(resultFile,'r') as f:
        lists = f.readlines()

        for i in range(0,len(lists)):

            list = lists[i].rstrip('\n')
            list = list.split(' ')

            labels.append(float(list[0]))
            index = int(list[0])+1
            pred_possible.append(float(list[index]))

            labels_class.append(int(list[0]))
            if float(list[1]) > float(list[2]):
                pred_class.append(0)
            else:
                pred_class.append(1)


    # ======================= metrics ============================
    precision, recall, threshold = metrics.precision_recall_curve(labels, pred_possible)

    fpr, tpr, thresholds = metrics.roc_curve(labels, pred_possible)

    ROC = metrics.roc_auc_score(labels,pred_possible)  # 梯形块分割，建议使用
    AP = metrics.average_precision_score(labels, pred_possible)  # 小矩形块分割
    F1_score = metrics.f1_score(labels_class,pred_class)

    # ======================= PLoting =============================
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(recall, precision, label=f"AP = {AP:.2f}\nF1 = {F1_score:.2f}",
             linewidth=2, linestyle='-', color='r', marker='o')
    plt.fill_between(recall, y1=precision, y2=0, step=None, alpha=0.2, color='b')
    plt.title("PR-Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.legend()

    plt.subplot(2, 2, 2)

    plt.plot(fpr, tpr, label=f"ROC_AUC = {ROC:.2f}\nF1 = {F1_score:.2f}",
             linewidth=2, linestyle='-', color='r', marker='o')
    plt.fill_between(fpr, y1=tpr, y2=0, step=None, alpha=0.2, color='b')
    plt.title("ROC-Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0, 1.05])
    plt.legend()

    plt.show()


def create_action_result():
    # 加载测试数据
    test_dataset = PoseDataSet(r'D:\code_data\human_pose', False)
    test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    net = NetV2()
    # 加载已训练的数据
    net.load_state_dict(torch.load("D:/py/openpose_lightweight/action_detect/checkPoint/action.pt"))
    net.to(DEVICE)  # 使用GPU进行训练

    labels = []
    pred = []

    with open('D:/py/openpose_lightweight/performance_evaluation/action_result.txt', 'a') as f:
        for i, (imgs, tags) in enumerate(test_dataLoader):
            # 测试集添加到GPU
            imgs = imgs.to(DEVICE)

            net.eval()  # 标明在测试环境下

            test_y = net(imgs)
            test_y = test_y.cpu().detach()[0].numpy()
            tags = torch.argmax(tags).numpy()
            labels.append(tags)
            pred.append(test_y[tags])

            f.write(f'{tags} {test_y[0]} {test_y[1]}\n')
            f.flush()





if __name__ == '__main__':
    # create_action_result()
    plot_action_AP('D:/py/openpose_lightweight/performance_evaluation/action_result.txt')






