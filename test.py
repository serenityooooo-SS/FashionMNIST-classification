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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, auc, confusion_matrix

test_set=torchvision.datasets.FashionMNIST("data",train=False,download=True,transform=torchvision.transforms.ToTensor())# ctrl+p可以看函数所需参数
test_loader=DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)


def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float().squeeze(1).squeeze(1).squeeze(1)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def norms_linf_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]


def blind_walk(model, X, y, args):
    # This function generates a single feature for the embedding matrix via Blind Walk Attack
    start = time.time()
    # params for noise magnitudes
    uni, std, scale = (0.005, 0.005, 0.01)
    num_steps=50
    is_training = model.training
    # Gaussian Noise Sampler
    noise_2 = lambda X: torch.normal(0, std, size=X.shape)
    # Laplacian Noise Sampler
    noise_1 = lambda X: torch.from_numpy(np.random.laplace(loc=0.0, scale=scale, size=X.shape)).float().to(X.device)
    # Uniform Noise Sampler
    noise_inf = lambda X: torch.empty_like(X).uniform_(-uni, uni)

    noise_map = {"l1": noise_inf, "l2": noise_2, "linf": noise_inf}
    # initialize at x+\delta_p
    mag = 1

    delta = noise_map[args.distance](X)
    delta_base = delta.clone()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)
    with torch.no_grad():  # gradient free
        for t in range(num_steps):
            if t > 0:
                preds = model(X_r + delta_r)
                new_remaining = (preds.max(1)[1] == y[remaining])
                c_remaining=remaining.clone()
                c_remaining[remaining]= new_remaining
                remaining=c_remaining

                #remaining[remaining] = new_remaining
            else:
                preds = model(X + delta)
                remaining = (preds.max(1)[1] == y)

            if remaining.sum() == 0: break

            # Only query the data points that have still not reached their neighbour (save queries :)
            X_r = X[remaining];
            delta_r = delta[remaining]
            preds = model(X_r + delta_r)
            ## Move by one more step in the same initial direction for the points still in their true class
            mag += 1;
            delta_r = delta_base[remaining] * mag
            # clip X+delta_r[remaining] to [0,1]
            delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1 - X_r)
            delta[remaining] = delta_r.detach()

        print(
            f"Number of steps = {t + 1} | Failed to convert = {(model(X + delta).max(1)[1] == y).sum().item()} | Time taken = {time.time() - start}")
    if is_training:
        model.train()
    return delta


def get_label_only_blind_walk_embeddings(args, loader, model, num_images=1000):
    ## This function is used to create the output embedding (size 30) via the blind walk attack
    print("Getting Blind Walk Embeddings")
    batch_size = args.batch_size
    max_iter = num_images / batch_size
    lp_dist = [[], [], []]
    for i, batch in enumerate(loader):
        for j, distance in enumerate(["linf", "l2", "l1"]):
            # linf distance corresponds to uniform noise
            # l2 distance corresponds to gaussian noise
            # l1 distance corresponds to laplacian noise
            temp_list = []
            for target_i in range(10):
                # 10 random starts for 10*3 = 30 size embedding
                X, y = batch[0].to(device), batch[1].to(device)
                args.distance = distance
                preds = model(X)
                delta = blind_walk(model, X, y, args)  # get one perturbation via blind walk
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)  ## Distance of perturbation is the feature in the embedding
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            temp_dist = torch.cat(temp_list, dim=1)
            lp_dist[j].append(temp_dist)
        if i + 1 >= max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim=-1);
    print(full_d.shape)
    print(full_d)

    return full_d

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


model=torch.load("./models/cnn_267.pth", map_location=torch.device("cpu"))

loss_fn=nn.CrossEntropyLoss()

model.eval()# 对特定的层有作用,例如dropout
total_test_loss=0
total_accurary=0
total_precise=0
total_recall = 0
y_true=torch.Tensor()
y_pred=torch.Tensor()
y_score=torch.Tensor()
y_one_hot=torch.Tensor()

with torch.no_grad():# 设置梯度都没有，不需优化仅仅测试
    for data in test_loader:
        imgs,targets=data
        outputs=model(imgs)
        loss=loss_fn(outputs,targets)
        total_test_loss=total_test_loss+loss
        accurary=(outputs.argmax(1)==targets).sum()
        total_accurary=total_accurary+accurary
        precise = precision_score(targets,outputs.argmax(1),average='weighted')
        total_precise=total_precise+precise
        recall=recall_score(targets,outputs.argmax(1),average='weighted')
        total_recall = total_recall + recall
        #print(outputs)
        #print(outputs.argmax(1))

        y_true = torch.cat((y_true, targets), 0)
        #print("y_true:", y_true)
        #print("y_true.shape:", y_true.shape)
        y_pred = torch.cat((y_pred, outputs.argmax(1)), 0)
        #print("y_pred:", y_pred)
        #print("y_pred.shape:", y_pred.shape)
        y_score = torch.cat((y_score, outputs), 0)
        #print("y_score:", y_score)
        #print("y_score.shape:", y_score.shape)  # 样本量*类别数，值为概率
        y_one_hot = label_binarize(y_true.cpu(), np.arange(10))  # 转换为类似于2进制编码
        #print("y_one_hot:", y_one_hot)




    precise = precision_score(y_true.cpu(), y_pred.cpu(), average='weighted')
    recall = recall_score(y_true.cpu(), y_pred.cpu(), average='weighted')
    f1score = f1_score(y_true.cpu(), y_pred.cpu(), average='weighted')
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score.cpu().ravel())
    print('调用函数auc：', roc_auc_score(y_one_hot, y_score.cpu(), average='micro'))
    #print("fpr:{},tpr:{},thresholds:{}".format(fpr, tpr, thresholds))
    print("整体测试集的loss:{}".format(total_test_loss))
    print("整体测试集的accurary:{}".format(total_accurary/len(test_set)))
    print("整体测试集的precise:{}".format(precise))
    print("整体测试集的recall:{}".format(recall))
    print("整体测试集的f1-score:{}".format(f1score))
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from matplotlib import pyplot as plt

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    confusion_mat = confusion_matrix(y_true, y_pred)
    print("confusion_mat.shape : {}".format(confusion_mat.shape))
    print("confusion_mat : {}".format(confusion_mat))

    # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    disp.plot(
        include_values=True,  # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",  # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,  # 同上
        xticks_rotation="horizontal",  # 同上
        values_format="d"  # 显示的数值格式
    )
    plt.show()

    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    norm_conf_mx = confusion_mat / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()


    #绘制ROC-AUC曲线图
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], y_score.cpu()[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_one_hot.ravel(), y_score.cpu().ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(10):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= 10
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial Training')
    ## Basics
    parser.add_argument("--batch_size", help="Batch Size for Train Set (Default = 100)", type=int, default=100)
    parser.add_argument("--distance", help="Type of Adversarial Perturbation",
                        type=str)  # , choices = ["linf", "l1", "l2", "vanilla"])
    return parser

parser = parse_args()
args = parser.parse_args()
args.batch_size=64
args.distance='linf'
device=torch.device("cuda:{0}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

get_label_only_blind_walk_embeddings(args=args,loader=test_loader,model=model)

