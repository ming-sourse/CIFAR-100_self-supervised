import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import random

# 设置随机种子
seed = 42
torch.manual_seed(seed)
random.seed(seed)

# 数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载STL-10数据集 (使用无标签的部分)
full_dataset = datasets.STL10(root='./data_STL', split='unlabeled', download=True, transform=transform)

def get_subset(dataset, num_samples):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:num_samples]
    return torch.utils.data.Subset(dataset, subset_indices)

# 准备小规模数据集
small_dataset = get_subset(full_dataset, 10000)  # 使用10000张图片

# 数据加载器
small_loader = DataLoader(small_dataset, batch_size=128, shuffle=True, num_workers=4)
full_loader = DataLoader(full_dataset, batch_size=128, shuffle=True, num_workers=4)

class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLR, self).__init__()
        self.backbone = base_model(pretrained=False)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 去掉最后一个全连接层

        # 获取最后一个卷积层的输出特征数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 96, 96)
            dummy_output = self.backbone(dummy_input)
            last_conv_output_dim = dummy_output.shape[1]

        self.projection_head = nn.Sequential(
            nn.Linear(last_conv_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x).squeeze()
        z = self.projection_head(h)
        return h, z

class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        target = torch.arange(batch_size * 2, device=z_i.device) % (batch_size * 2 - 1)

        loss = self.criterion(sim_matrix, target)
        loss /= (2 * batch_size)
        return loss

def pretrain_model(loader, model, optimizer, num_epochs, save_path, log_dir):
    criterion = NTXentLoss(temperature=0.5)
    writer = SummaryWriter(log_dir)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(loader):
            images = torch.cat([images, images], dim=0)
            images = images.cuda()
            optimizer.zero_grad()

            _, z = model(images)
            z_i, z_j = z[:images.size(0) // 2], z[images.size(0) // 2:]

            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss}")
        writer.add_scalar('Loss/train', avg_loss, epoch)

    writer.close()
    torch.save(model.state_dict(), save_path)

# 预训练小规模数据集模型
model_small = SimCLR(resnet18, 128).cuda()
optimizer_small = optim.Adam(model_small.parameters(), lr=3e-4)
pretrain_model(small_loader, model_small, optimizer_small, 20, 'simclr_resnet18_small.pth', 'runs/small')

# 预训练大规模数据集模型
model_full = SimCLR(resnet18, 128).cuda()
optimizer_full = optim.Adam(model_full.parameters(), lr=3e-4)
pretrain_model(full_loader, model_full, optimizer_full, 20, 'simclr_resnet18_full.pth', 'runs/full')
