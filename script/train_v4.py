"""GeneExpressTransformerV4 训练脚本。

支持消融实验，通过参数控制特征组合：
- baseline: promoter + halflife (等价 v3)
- +seq: baseline + 序列内在特征 (GC/CpG)
- +encode: baseline + ENCODE 表观信号
- +all: baseline + 序列特征 + ENCODE
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.modelv4 import GeneExpressTransformerV4
from utils.augment import augment_promoter
from utils.feature_engineering import compute_sequence_features, SEQ_FEAT_DIM


class V4Dataset(Dataset):
    """v4 数据集，支持多种特征组合。"""

    def __init__(
        self,
        promoters: torch.Tensor,
        halflifes: torch.Tensor,
        labels: torch.Tensor,
        encode_signals: torch.Tensor | None = None,
        seq_features: torch.Tensor | None = None,
        use_seq_feat: bool = False,
        augment: bool = True,
    ):
        self.promoters = promoters
        self.halflifes = halflifes
        self.labels = labels
        self.encode_signals = encode_signals
        self.seq_features = seq_features
        self.use_seq_feat = use_seq_feat
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p = self.promoters[idx]
        h = self.halflifes[idx]
        l = self.labels[idx]

        if self.augment:
            p = augment_promoter(p)

        result = [p, h, l]

        if self.encode_signals is not None:
            result.append(self.encode_signals[idx])

        if self.seq_features is not None:
            result.append(self.seq_features[idx])

        return result


# 终端颜色
class C:
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


def load_epigenomic(path: Path, split: str) -> torch.Tensor | None:
    """加载预计算的表观特征。"""
    if not path.exists():
        return None
    data = torch.load(path, weights_only=False)
    signals = data.get(split)
    if signals is not None:
        return torch.tensor(signals, dtype=torch.float32)
    return None


def precompute_seq_features(promoters: torch.Tensor) -> torch.Tensor:
    """预计算序列内在特征（验证/测试集用，无增强）。"""
    return compute_sequence_features(promoters)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    has_encode=False, has_seq=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and device.type == "cuda"

    for batch in loader:
        if has_encode and has_seq:
            p, h, labels, enc, sf = batch
        elif has_encode:
            p, h, labels, enc = batch
            sf = None
        elif has_seq:
            p, h, labels, sf = batch
            enc = None
        else:
            p, h, labels = batch
            enc, sf = None, None

        p = p.to(device)
        h = h.to(device)
        labels = labels.to(device)
        enc = enc.to(device) if enc is not None else None
        sf = sf.to(device) if sf is not None else None

        optimizer.zero_grad()

        def fwd():
            return model(p, h, encode_signal=enc, seq_features=sf)

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = fwd()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = fwd()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, has_encode=False, has_seq=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        if has_encode and has_seq:
            p, h, labels, enc, sf = batch
        elif has_encode:
            p, h, labels, enc = batch
            sf = None
        elif has_seq:
            p, h, labels, sf = batch
            enc = None
        else:
            p, h, labels = batch
            enc, sf = None, None

        p = p.to(device)
        h = h.to(device)
        labels = labels.to(device)
        enc = enc.to(device) if enc is not None else None
        sf = sf.to(device) if sf is not None else None

        outputs = model(p, h, encode_signal=enc, seq_features=sf)
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
    parser = argparse.ArgumentParser(description="GeneExpressTransformerV4 训练")
    parser.add_argument("--no-augment", action="store_true", help="禁用数据增强")
    parser.add_argument("--no-tta", action="store_true", help="禁用 TTA")
    parser.add_argument(
        "--features",
        choices=["baseline", "seq", "encode", "all"],
        default="baseline",
        help="特征组合: baseline/seq/encode/all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_augment = not args.no_augment
    use_tta = not args.no_tta
    feat_mode = args.features

    use_seq_feat = feat_mode in ("seq", "all")
    use_encode = feat_mode in ("encode", "all")

    seed_everything(42)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(C.info(f"设备: {device}"))
    print(C.info(f"特征模式: {C.bold(feat_mode)}"))
    feats_desc = ["promoter", "halflife"]
    if use_seq_feat:
        feats_desc.append("序列特征(GC+CpG)")
    if use_encode:
        feats_desc.append("ENCODE(H3K4me3+H3K27ac+DNase)")
    print(f"  特征: {', '.join(feats_desc)}")

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

    # 加载 epigenomic 数据
    train_enc, valid_enc, test_enc = None, None, None
    if use_encode:
        epi_path = data_dir / "epigenomic.pt"
        if epi_path.exists():
            train_enc = load_epigenomic(epi_path, "train")
            valid_enc = load_epigenomic(epi_path, "valid")
            test_enc = load_epigenomic(epi_path, "test")
            shape_str = f"{train_enc.shape}" if train_enc is not None else "N/A"
            print(C.ok("ENCODE 数据加载完成") + f"  形状: {shape_str}")
        else:
            print(C.warn("ENCODE 数据不存在，请先运行: python script/prepare_epigenomic.py"))
            use_encode = False

    # 加载预计算的序列特征
    train_sf, valid_sf, test_sf = None, None, None
    if use_seq_feat:
        sf_train_path = data_dir / "seq_features_train.pt"
        if sf_train_path.exists():
            print(C.dim("加载预计算序列特征..."))
            train_sf = torch.load(data_dir / "seq_features_train.pt", weights_only=True)
            valid_sf = torch.load(data_dir / "seq_features_valid.pt", weights_only=True)
            test_sf = torch.load(data_dir / "seq_features_test.pt", weights_only=True)
            print(C.ok("序列特征加载完成") + f"  维度: {train_sf.shape[1]}")
        else:
            print(C.warn("序列特征不存在，请先运行: python script/precompute_seq_features.py"))
            use_seq_feat = False

    # 构建 DataLoader
    batch_size = 32
    train_ds = V4Dataset(
        train_p, train_h, train_l,
        encode_signals=train_enc,
        seq_features=train_sf,
        use_seq_feat=use_seq_feat,
        augment=use_augment,
    )
    valid_ds = V4Dataset(
        valid_p, valid_h, valid_l,
        encode_signals=valid_enc,
        seq_features=valid_sf,
        use_seq_feat=use_seq_feat,
        augment=False,
    )
    test_ds = V4Dataset(
        test_p, test_h, test_l,
        encode_signals=test_enc,
        seq_features=test_sf,
        use_seq_feat=use_seq_feat,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 模型
    encode_channels = 3 if use_encode else 0
    seq_dim = SEQ_FEAT_DIM if use_seq_feat else 0
    model = GeneExpressTransformerV4(
        encode_channels=encode_channels,
        seq_feat_dim=seq_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {C.hi('GeneExpressTransformerV4')}  参数量: {C.bold(f'{total_params:,}')}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    num_epochs = 35 if use_augment else 25
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    best_valid_loss = float("inf")
    best_epoch = 0
    patience = 8
    no_improve = 0

    aug_tag = C.ok("开启") if use_augment else C.warn("关闭")
    tta_tag = C.ok("开启") if use_tta else C.warn("关闭")
    amp_tag = C.ok("开启") if scaler else C.warn("关闭")
    print(f"数据增强: {aug_tag}  TTA: {tta_tag}  Batch: {C.bold(batch_size)}  AMP: {amp_tag}")
    print(C.info(f"开始训练  {num_epochs} epochs  patience {patience}"))

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            has_encode=use_encode, has_seq=use_seq_feat,
        )
        valid_loss, valid_acc, valid_auc, valid_f1 = evaluate(
            model, valid_loader, criterion, device,
            has_encode=use_encode, has_seq=use_seq_feat,
        )
        scheduler.step(valid_loss)
        lr = optimizer.param_groups[0]["lr"]

        saved = ""
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            save_name = f"modelv4_{feat_mode}_best.pt"
            torch.save(model.state_dict(), data_dir / save_name)
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

    # 测试评估
    save_name = f"modelv4_{feat_mode}_best.pt"
    print(C.info(f"加载最佳模型 {save_name}，测试集评估"))
    model.load_state_dict(torch.load(data_dir / save_name, weights_only=True))
    model.eval()

    all_probs, all_labels_tta = [], []

    with torch.no_grad():
        for batch in test_loader:
            if use_encode and use_seq_feat:
                p, h, labels, enc, sf = batch
            elif use_encode:
                p, h, labels, enc = batch
                sf = None
            elif use_seq_feat:
                p, h, labels, sf = batch
                enc = None
            else:
                p, h, labels = batch
                enc, sf = None, None

            p = p.to(device)
            h = h.to(device)
            enc = enc.to(device) if enc is not None else None
            sf = sf.to(device) if sf is not None else None

            logits_fwd = model(p, h, encode_signal=enc, seq_features=sf)

            if use_tta:
                p_rc = p.flip(1).flip(2)
                if use_seq_feat:
                    sf_rc = compute_sequence_features(p_rc.cpu()).to(device)
                else:
                    sf_rc = None
                logits_rc = model(p_rc, h, encode_signal=enc, seq_features=sf_rc)
                probs = torch.softmax(logits_fwd + logits_rc, dim=1)
            else:
                probs = torch.softmax(logits_fwd, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels_tta.extend(labels.numpy())

    tta_probs = np.array(all_probs)
    tta_labels = np.array(all_labels_tta)
    tta_preds = tta_probs.argmax(1)
    test_acc = accuracy_score(tta_labels, tta_preds)
    test_auc = roc_auc_score(tta_labels, tta_preds)
    test_f1 = f1_score(tta_labels, tta_preds)

    print(
        C.ok("测试完成")
        + f"  Acc {C.bold(f'{test_acc:.4f}')}  "
        f"AUC {C.bold(f'{test_auc:.4f}')}  "
        f"F1 {C.bold(f'{test_f1:.4f}')}"
    )

    # 记录实验日志
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    csv_path = log_dir / "experiments.csv"
    header = [
        "timestamp", "model", "features", "augment", "tta", "amp",
        "batch_size", "lr", "weight_decay", "scheduler",
        "best_epoch", "final_epoch", "num_epochs",
        "test_acc", "test_auc", "test_f1", "params",
    ]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "v4", feat_mode, use_augment, use_tta, scaler is not None,
        batch_size, 5e-5, 1e-4, "ReduceLROnPlateau",
        best_epoch, epoch, num_epochs,
        f"{test_acc:.4f}", f"{test_auc:.4f}", f"{test_f1:.4f}",
        total_params,
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(C.ok(f"实验日志已保存到 {csv_path}"))


if __name__ == "__main__":
    main()
