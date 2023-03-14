import glob
import random
import os
from sklearn.metrics import precision_recall_curve
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# gpu setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

def seed_torch(seed=4):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




class CS4243_dataset(Dataset):

    def __init__(self, root_path, mode='train', transform=None, L = None):

        self.transform = transform
        self.root_path = root_path
        self.mode = mode
        self.L = L

        self.img_path_list = sorted(glob.glob(f"{self.root_path}/*.jpg"))
        self.labels = 'bag_data/empt_arr.txt'
        self.labels = np.loadtxt(self.labels).astype(np.int)


    def __getitem__(self, index):
        index +=  self.L
        # index = 102

        N_img_name = self.img_path_list[index]
        N_image = Image.open(N_img_name)

        first2N_L_img_list = []
        compare_index_list = []
        for i in range(index - self.L + 1):
            first2N_L_img = self.img_path_list[i]
            first2N_L_img = Image.open(first2N_L_img)
            first2N_L_img = self.transform(first2N_L_img)
            first2N_L_img_list.append(first2N_L_img)
            compare_index_list.append(i)

        compare_index_list = torch.tensor(np.array(compare_index_list, dtype=np.int))

        first2N_L_img_tensor = torch.stack(first2N_L_img_list, dim=0)

        target = torch.tensor(self.labels[index,:])
        target = target[compare_index_list]

        N_image = self.transform(N_image)


        return [N_image,first2N_L_img_tensor, target]

    def __len__(self):
        return len(self.img_path_list)



# val/test a Epoch
def infer(model, test_loader):
    model.eval()  # close BN and dropout layer
    print('===== Validation =====')

    with torch.no_grad():
        label_list = []
        cos_sim_list = []

        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            if i + L >= len(test_loader) - 1 :
                break

            X_train, sequence ,label = data
            X_train = X_train.to(device)
            sequence = sequence.to(device)
            label = label.to(device).long()

            X_embed = net(X_train)

            seq_embed = net(sequence.squeeze(0))

            cos_sim = torch.cosine_similarity(X_embed, seq_embed, dim=1)

            del X_embed,seq_embed

            label_list.extend(label.cpu().numpy().flatten().astype(np.float32))
            cos_sim_list.extend(cos_sim.cpu().numpy().flatten())

        precision, recall, thresholds = precision_recall_curve(label_list, cos_sim_list)
        plt.figure(1)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig('p-r.png')
        plt.show()
        # precision , recall ,thresholds = precision_recall_curve(label.cpu().numpy().flatten().astype(np.float32), cos_sim.cpu().numpy().flatten())




if __name__ == '__main__':


    seed_torch()

    img_names = []  # ID
    targets = []  # labels

    root_path = 'bag_data/cam1'  # locally

    """#### Image preporcess: transform"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    resize_H, resized_W = 224, 224  #
    resize = transforms.Resize([resize_H, resized_W])

    transformations = transforms.Compose([
        resize,
        transforms.ToTensor(),  # Tturn gray level from 0-255 into 0-1
        normalize
    ])  # change 0-1 into (-1, 1)

    L = 1000

    batch_size = 1

    train_dataset = CS4243_dataset(root_path, mode='train', transform=transformations,L=L)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)


    """### initalize Model & Hyper params"""
    from torchvision import models

    net = models.vgg16(pretrained=True).cuda()
    net.eval()

    # inference

    infer(net, train_loader)