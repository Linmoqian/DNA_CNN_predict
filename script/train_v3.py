"""GeneExpressTransformer 训练脚本。"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.modelv3 import GeneExpressTransformer
from utils.augment import AugmentedDataset


# 终端颜色工具
class C:
    """语义化颜色: 绿=成功 黄=警告 红=错误 青=提示 蓝=高亮 灰=次要"""

    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RST = "\033[0m"

    @staticmethod
    def ok(msg):
        return f"{C.GREEN}{msg}{C.RST}"

    @staticmethod
    def warn(msg):
        return f"{C.YELLOW}{msg}{C.RST}"

    @staticmethod
    def err(msg):
        return f"{C.RED}{msg}{C.RST}"

    @staticmethod
    def info(msg):
        return f"{C.CYAN}{msg}{C.RST}"

    @staticmethod
    def hi(msg):
        return f"{C.BLUE}{msg}{C.RST}"

    @staticmethod
    def dim(msg):
        return f"{C.GRAY}{msg}{C.RST}"

    @staticmethod
    def bold(msg):
        return f"{C.BOLD}{msg}{C.RST}"


def load_hdf5(file_path: str):
    with h5py.File(file_path, "r") as f:
        halflife = torch.tensor(np.array(f["halflife"]), dtype=torch.float32)
        promoter = torch.tensor(np.array(f["promoter"]), dtype=torch.float32)
        labels = torch.tensor(np.array(f["label"]), dtype=torch.long)
    return promoter, halflife, labels


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for promoter, halflife, labels in loader:
        promoter, halflife, labels = (
            promoter.to(device),
            halflife.to(device),
            labels.to(device),
        )
        outputs = model(promoter, halflife)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for promoter, halflife, labels in loader:
        promoter, halflife, labels = (
            promoter.to(device),
            halflife.to(device),
            labels.to(device),
        )
        outputs = model(promoter, halflife)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / total, acc, auc, f1


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="GeneExpressTransformer 训练")
    parser.add_argument("--no-augment", action="store_true", help="禁用数据增强")
    return parser.parse_args()


def main():
    args = parse_args()
    use_augment = not args.no_augment

    seed_everything(42)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(C.info(f"设备: {device}"))

    # 加载数据
    print(C.dim("加载数据..."))
    train_p, train_h, train_l = load_hdf5(str(data_dir / "train.h5"))
    valid_p, valid_h, valid_l = load_hdf5(str(data_dir / "valid.h5"))
    test_p, test_h, test_l = load_hdf5(str(data_dir / "test.h5"))
    print(
        C.ok("数据加载完成")
        + f"  训练 {C.bold(len(train_l))}  "
        f"验证 {C.bold(len(valid_l))}  "
        f"测试 {C.bold(len(test_l))}"
    )

    train_ds = AugmentedDataset(train_p, train_h, train_l, augment=use_augment)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_p, valid_h, valid_l), batch_size=32)
    test_loader = DataLoader(TensorDataset(test_p, test_h, test_l), batch_size=32)
    aug_tag = C.ok("开启") if use_augment else C.warn("关闭")
    print(f"数据增强: {aug_tag}  (反向互补 + 随机平移 + 随机遮蔽)")

    # 模型
    model = GeneExpressTransformer(num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {C.hi('GeneExpressTransformer')}  参数量: {C.bold(f'{total_params:,}')}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 训练
    num_epochs = 35 if use_augment else 25
    best_valid_loss = float("inf")
    patience = 8
    no_improve = 0
    print(C.info(f"开始训练  共 {num_epochs} epochs  early stopping patience {patience}"))

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        valid_loss, valid_acc, valid_auc, valid_f1 = evaluate(
            model, valid_loader, criterion, device
        )
        scheduler.step(valid_loss)
        lr = optimizer.param_groups[0]["lr"]

        saved = ""
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), data_dir / "modelv3_best.pt")
            saved = f"  {C.ok('saved')}"
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"  {C.dim(f'Epoch {epoch:>2d}/{num_epochs}')}  "
            f"{C.hi(f'LR {lr:.1e}')}  "
            f"Train L {train_loss:.4f} A {train_acc:.4f}  "
            f"Valid L {valid_loss:.4f} A {valid_acc:.4f} "
            f"{C.bold(f'AUC {valid_auc:.4f}')}  "
            f"F1 {valid_f1:.4f}"
            f"{saved}"
        )

        if no_improve >= patience:
            print(C.warn(f"验证 Loss 连续 {patience} 轮未改善，提前停止"))
            break

    # 测试
    print(C.info("加载最佳模型，测试集评估"))
    model.load_state_dict(torch.load(data_dir / "modelv3_best.pt", weights_only=True))
    test_loss, test_acc, test_auc, test_f1 = evaluate(
        model, test_loader, criterion, device
    )
    print(
        C.ok("测试完成")
        + f"  Loss {test_loss:.4f}  "
        f"Acc {C.bold(f'{test_acc:.4f}')}  "
        f"AUC {C.bold(f'{test_auc:.4f}')}  "
        f"F1 {C.bold(f'{test_f1:.4f}')}"
    )


if __name__ == "__main__":
    main()
