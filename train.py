# 网络模型
# 损失函数
# 数据（输入，标注）
# cuda()


# 构建一个CNN网络实现分类器
from itertools import cycle

import numpy as np
import torch
import torchvision
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
import time
# 导入数据
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score,auc


train_set=torchvision.datasets.FashionMNIST("data",train=True,download=True,transform=torchvision.transforms.ToTensor())# ctrl+p可以看函数所需参数
train_set,val_set=torch.utils.data.random_split(train_set,[50000,10000])# 将60000的数据集随机划分为50000训练，10000验证

train_loader=DataLoader(dataset=train_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
val_loader=DataLoader(dataset=val_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

test_set=torchvision.datasets.FashionMNIST("data",train=False,download=True,transform=torchvision.transforms.ToTensor())# ctrl+p可以看函数所需参数
test_loader=DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

print("训练数据集的长度为：", len(train_set))
print("验证数据集的长度为：", len(val_set))
print("测试数据集的长度为：", len(test_set))

# 搭建神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
#初始化
def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

# 创建网络模型
cnn=CNN()
if torch.cuda.is_available():
    cnn=cnn.cuda()
cnn.apply(init_weights)

# 创建损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()

# 优化器
optimizier=torch.optim.Adam(cnn.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False)

# 设置网络参数
total_train_step=0
total_val_step=0
epoch = 1000


# 添加tensoboard
writer=SummaryWriter("logs_cnngpu")
start_time=time.time()
for i in range(epoch):
    print("--------------------------------------------")
    print("第{}轮训练开始：".format(i+1))

    # 开始训练
    cnn.train()# 对特定的层有作用,例如dropout
    for data in train_loader:
        imgs, targets=data
        # imgs=imgs.cuda()
        # targets=targets.cuda()
        outputs=cnn(imgs)
        loss=loss_fn(outputs,targets)

        optimizier.zero_grad()
        loss.backward()
        optimizier.step()

        total_train_step=total_train_step+1
        if total_train_step % 100==0:
            print("训练次数：{}，Loss：{}".format(str(total_train_step),str(loss.item())))
            writer.add_scalar("train_loss:",loss.item(),total_train_step)


    # 进行数据交叉验证
    cnn.eval()  # 对特定的层有作用,例如dropout
    total_val_loss = 0
    total_accurary = 0
    # y_true=torch.Tensor().cuda()
    # y_pred=torch.Tensor().cuda()
    # y_score=torch.Tensor().cuda()
    # y_one_hot=torch.Tensor().cuda()
    y_true = torch.Tensor()
    y_pred = torch.Tensor()
    y_score = torch.Tensor()
    y_one_hot = torch.Tensor()
    with torch.no_grad():# 设置梯度都没有，不需优化仅仅测试
        for data in val_loader:
            imgs,targets=data
            # imgs = imgs.cuda()
            # targets = targets.cuda()
            outputs=cnn(imgs)
            loss=loss_fn(outputs,targets)
            total_val_loss=total_val_loss+loss
            #print("预测值：",outputs.argmax(1))
            #print("实际值：",targets)
            accurary=(outputs.argmax(1)==targets).sum()
            #print("预测值=实际值的数目：",accurary)
            total_accurary=total_accurary+accurary
            y_true=torch.cat((y_true, targets), 0)
    #         print("y_true:",y_true)
    #         print("y_true.shape:",y_true.shape)
    #         y_pred=torch.cat((y_pred,outputs.argmax(1)),0)
    #         print("y_pred:",y_pred)
    #         print("y_pred.shape:", y_pred.shape)
    #         y_score=torch.cat((y_score,outputs),0)
    #         print("y_score:", y_score)
    #         print("y_score.shape:", y_score.shape)# 样本量*类别数，值为概率
    #         y_one_hot=label_binarize(y_true.cpu(), np.arange(10))#转换为类似于2进制编码
    #         print("y_one_hot:",y_one_hot)
    # precise = precision_score(y_true.cpu(), y_pred.cpu(), average='weighted')
    # recall = recall_score(y_true.cpu(), y_pred.cpu(), average='weighted')
    # f1score=f1_score(y_true.cpu(), y_pred.cpu(), average='weighted')
    # fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score.cpu().ravel())
    # print('调用函数auc：', roc_auc_score(y_one_hot, y_score.cpu(), average='micro'))
    # print("fpr:{},tpr:{},thresholds:{}".format(fpr, tpr, thresholds))
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(10):
    #     fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], y_score.cpu()[:, i] )
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_one_hot.ravel(), y_score.cpu().ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # # Compute macro-average ROC curve and ROC area（方法一）
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(10):
    #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # # Finally average it and compute AUC
    # mean_tpr /= 10
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # lw = 2
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(10), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

    print("整体验证集的loss:{}".format(total_val_loss))
    print("整体验证集的accurary:{}".format(total_accurary / len(val_set)))# 总准确个数/总个数
    #print("整体验证集的precise:{}".format(precise))
    #print("整体验证集的recall:{}".format(recall))
    #print("整体验证集的f1-score:{}".format(f1score))

    # 绘图
    # auc = auc(fpr, tpr)
    # mpl.rcParams['font.sans-serif'] = u'SimHei'
    # mpl.rcParams['axes.unicode_minus'] = False
    # # FPR就是横坐标,TPR就是纵坐标
    # plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
    # plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    # plt.xlim((-0.01, 1.02))
    # plt.ylim((-0.01, 1.02))
    # plt.xticks(np.arange(0, 1.1, 0.1))
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xlabel('False Positive Rate', fontsize=13)
    # plt.ylabel('True Positive Rate', fontsize=13)
    # plt.grid(b=True, ls=':')
    # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    # plt.title(u'大类问题一分类后的ROC和AUC', fontsize=17)
    # plt.show()

    writer.add_scalar("val_loss:", loss.item(), total_val_step)
    writer.add_scalar("accurary:", (total_accurary / len(val_set)).item(), total_val_step)
    #writer.add_scalar("val_precise:", precise.item(), total_val_step)
    #writer.add_scalar("val_recall:", recall.item(), total_val_step)
    total_val_step = total_val_step+1

    torch.save(cnn,"models/cnn_{}.pth".format(i))
    # torch.save(cnn.state_dict(),"models/cnn_{}.pth".format(i))#官方推荐
    print("模型已保存为：cnn_{}.pth".format(i))
    end_time = time.time()
    print("训练时间为：", end_time - start_time)



writer.close()