import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

transform_test = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

# 加载CIFAR-100数据集
train_dataset = datasets.CIFAR100(root='/hy-tmp/', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='/hy-tmp/', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

class LinearClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(LinearClassifier, self).__init__()
        self.backbone = base_model(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 获取最后一个卷积层的输出特征数
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 96, 96)
            dummy_output = self.backbone(dummy_input)
            last_conv_output_dim = dummy_output.shape[1]

        self.fc = nn.Linear(last_conv_output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x).squeeze()
        logits = self.fc(h)
        return logits

# 使用预训练的ResNet-18
model = LinearClassifier(resnet18, 100).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

writer = SummaryWriter('runs/super_pre')

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / total
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)
    print(f'Train Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {acc}%')

def test(model, test_loader, criterion, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    acc = 100. * correct / total
    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)
    print(f'Test Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {acc}%')

num_epochs = 50
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion, epoch)

writer.close()