import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import h5py
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

#需要把每个特征通过一个通道

class ConvModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvModel, self).__init__()
        
        self.conv1_promoter = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.bn1_promoter = nn.BatchNorm1d(32)  # 批标准化
        self.conv2_promoter = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2_promoter = nn.BatchNorm1d(64)  # 批标准化
        self.pool_promoter = nn.MaxPool1d(kernel_size=2, stride=2)

        self.attention_fc = nn.Linear(64, 64)

        self.fc_halflife = nn.Linear(8, 32)  

        self.promoter_out_size = 64 * (20000 // 2 // 2)  
        
        self.fc1 = nn.Linear(self.promoter_out_size + 32, 128)  
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.8)

    def forward(self, x_promoter, x_halflife):
        # 启动子特征数据处理、批标准化
        x_promoter = x_promoter.permute(0, 2, 1) 
        x_promoter = F.relu(self.conv1_promoter(x_promoter))
        x_promoter = self.bn1_promoter(x_promoter)  
        x_promoter = self.pool_promoter(x_promoter)  
        x_promoter = F.relu(self.conv2_promoter(x_promoter))
        x_promoter = self.bn2_promoter(x_promoter)  
        x_promoter = self.pool_promoter(x_promoter)  

        # 引入注意力机制
        attention_weights = torch.sigmoid(self.attention_fc(x_promoter.permute(0, 2, 1))) 
        x_promoter = x_promoter * attention_weights.permute(0, 2, 1)  

        x_promoter = x_promoter.view(x_promoter.size(0), -1) 

        # 半衰期特征处理
        x_halflife = F.relu(self.fc_halflife(x_halflife))  

        # 合并并通过全连接层
        x = torch.cat((x_promoter, x_halflife), dim=1) 
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)
        out = self.fc2(x)

        return out

# 加载 HDF5 数据集
def load_hdf5(file_path):（
    with h5py.File(file_path, 'r') as f:
        gene_ids = list(f['gene_id'])
        halflife = torch.tensor(np.array(f['halflife']), dtype=torch.float32)
        promoter = torch.tensor(np.array(f['promoter']), dtype=torch.float32)
        labels = torch.tensor(np.array(f['label']), dtype=torch.long)
        
        return gene_ids, halflife, promoter, labels

#数据的初始化
def date_init():
    global train_gene_ids, train_half_lives, train_promoters, train_labels
    global valid_gene_ids, valid_half_lives, valid_promoters, valid_labels
    global test_gene_ids, test_half_lives, test_promoters, test_labels

    train_gene_ids, train_half_lives, train_promoters, train_labels = load_hdf5('train.h5')
    valid_gene_ids, valid_half_lives, valid_promoters, valid_labels = load_hdf5('valid.h5')
    test_gene_ids, test_half_lives, test_promoters, test_labels = load_hdf5('test.h5')

def create_dataloader(promoters, halflife, labels, batch_size=32):
    dataset = TensorDataset(promoters, halflife, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for promoters, halflife, labels in train_loader:
        promoters, halflife, labels = promoters.to(device), halflife.to(device), labels.to(device)

        outputs = model(promoters, halflife)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc

def validate_model(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for promoters, halflife, labels in valid_loader:
            promoters, halflife, labels = promoters.to(device), halflife.to(device), labels.to(device)

            outputs = model(promoters, halflife)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    valid_loss = running_loss / len(valid_loader)
    valid_acc = 100.0 * correct / total

    return valid_loss, valid_acc

# 在验证和测试时计算评估指标
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for promoters, halflife, labels in data_loader:
            promoters, halflife, labels = promoters.to(device), halflife.to(device), labels.to(device)
            
            outputs = model(promoters, halflife)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return accuracy, auc, f1

# 绘制训练过程中的损失和准确率
def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Valid Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, valid_accuracies, label="Valid Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = get_device()
    date_init()
    
    num_classes = 2  
    model = ConvModel(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    train_loader = create_dataloader(train_promoters, train_half_lives, train_labels)
    valid_loader = create_dataloader(valid_promoters, valid_half_lives, valid_labels)
    test_loader = create_dataloader(test_promoters, test_half_lives, test_labels)

    num_epochs = 10

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = validate_model(model, valid_loader, criterion, device)

        scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

        # 计算验证集的准确率、AUC 和 F1分数
        valid_accuracy, valid_auc, valid_f1 = evaluate_model(model, valid_loader, device)
        print(f"Valid Accuracy: {valid_accuracy:.2f}%, AUC: {valid_auc:.4f}, F1 Score: {valid_f1:.4f}")
        print("-" * 20)

    # 绘制损失和准确率图
    plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies)

    # 计算测试集的准确率、AUC 和 F1分数
    test_accuracy, test_auc, test_f1 = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%, AUC: {test_auc:.4f}, F1 Score: {test_f1:.4f}')
