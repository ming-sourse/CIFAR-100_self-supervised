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

# 定义模型
model = resnet18(pretrained=False, num_classes=100).cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练和测试函数
def train(model, train_loader, criterion, optimizer, epoch, writer):
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

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')

    acc = 100. * correct / total
    writer.add_scalar('Loss/train', total_loss/len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)
    print(f'Train Loss: {total_loss/len(train_loader)}, Accuracy: {acc}%')

def test(model, test_loader, criterion, epoch, writer):
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

    acc = 100. * correct / total
    writer.add_scalar('Loss/test', total_loss/len(test_loader), epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)
    print(f'Test Loss: {total_loss/len(test_loader)}, Accuracy: {acc}%')

# 训练和测试
writer = SummaryWriter('runs/cifar100_supervised')
num_epochs = 50
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch, writer)
    test(model, test_loader, criterion, epoch, writer)

writer.close()
torch.save(model.state_dict(), 'cifar100_resnet18_supervised.pth')