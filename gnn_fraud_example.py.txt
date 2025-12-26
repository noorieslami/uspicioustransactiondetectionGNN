import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.loader import DataLoader


# -----------------------------
# 1) خواندن و آماده‌سازی داده
# -----------------------------
def load_transactions(csv_path: str):
    """
    csv_path: مسیر فایل transactions.csv
    """
    df = pd.read_csv(csv_path)

    # حذف ردیف‌های ناقص
    df = df.dropna(subset=["src_account_id", "dst_merchant_id", "label"])

    # تبدیل label به int (0/1)
    df["label"] = df["label"].astype(int)

    # اگر timestamp داری و خواستی بر اساس زمان split کنی، می‌توانی اینجا تبدیل به datetime کنی
    # df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def encode_ids(df: pd.DataFrame):
    """
    account_id و merchant_id را به اندیس عددی تبدیل می‌کند.
    """
    df = df.copy()

    acc_ids, acc_index = np.unique(df["src_account_id"], return_inverse=True)
    merch_ids, merch_index = np.unique(df["dst_merchant_id"], return_inverse=True)

    df["src_acc_idx"] = acc_index
    df["dst_merch_idx"] = merch_index

    num_accounts = len(acc_ids)
    num_merchants = len(merch_ids)
    return df, num_accounts, num_merchants


# -----------------------------
# 2) ساخت گراف ناهمگون با PyG
# -----------------------------
def build_hetero_graph(df: pd.DataFrame, num_accounts: int, num_merchants: int):
    """
    از روی DataFrame یک HeteroData می‌سازد.
    node types: "account", "merchant"
    edge type: ("account", "transacts", "merchant")
    """

    data = HeteroData()

    # ویژگی ساده برای هر نوع گره (اینجا فقط یک فیچر ثابت؛ برای کار واقعی از فیچرهای واقعی استفاده کن)
    data["account"].num_nodes = num_accounts
    data["merchant"].num_nodes = num_merchants

    # برای سادگی، یک feature اسکالر 1.0 برای همه گره‌ها می‌گذاریم
    data["account"].x = torch.ones((num_accounts, 1), dtype=torch.float)
    data["merchant"].x = torch.ones((num_merchants, 1), dtype=torch.float)

    # یال‌ها: account -> merchant
    src = torch.tensor(df["src_acc_idx"].values, dtype=torch.long)
    dst = torch.tensor(df["dst_merch_idx"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    data["account", "transacts", "merchant"].edge_index = edge_index

    # برچسب هر یال (تراکنش) = label
    data["account", "transacts", "merchant"].y = torch.tensor(df["label"].values, dtype=torch.float)

    # می‌توانیم edge features ساده هم اضافه کنیم (مثلاً مبلغ نرمال‌شده)
    if "amount" in df.columns:
        amount = df["amount"].values.astype(float)
        amount = (amount - amount.mean()) / (amount.std() + 1e-6)
        data["account", "transacts", "merchant"].edge_attr = torch.tensor(amount, dtype=torch.float).unsqueeze(-1)

    return data


# -----------------------------
# 3) مدل GNN ناهمگون
# -----------------------------
class HeteroGNN(nn.Module):
    """
    مدل ساده:
    - دو لایه HeteroConv (هر کدام با SAGEConv روی دو جهت گراف)
    - بردار account و merchant را برای هر یال می‌گیریم
    - concat می‌کنیم و از MLP می‌گذرانیم -> احتمال تقلب
    """

    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()

        # لایه اول
        self.conv1 = HeteroConv({
            ("account", "transacts", "merchant"): SAGEConv((in_channels, in_channels), hidden_channels),
            ("merchant", "rev_transacts", "account"): SAGEConv((in_channels, in_channels), hidden_channels),
        })

        # لایه دوم
        self.conv2 = HeteroConv({
            ("account", "transacts", "merchant"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ("merchant", "rev_transacts", "account"): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
        })

        # لایه نهایی برای پیش‌بینی روی یال (تراکنش)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data: HeteroData):
        x_dict = {
            "account": data["account"].x,
            "merchant": data["merchant"].x,
        }

        # ساخت یال‌های معکوس (merchant -> account) برای پیام‌رسانی دوطرفه
        edge_index_dict = {
            ("account", "transacts", "merchant"): data["account", "transacts", "merchant"].edge_index,
            ("merchant", "rev_transacts", "account"): data["account", "transacts", "merchant"].edge_index.flip(0),
        }

        # لایه اول
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}

        # لایه دوم
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}

        # پیش‌بینی روی یال‌های "transacts"
        src, dst = data["account", "transacts", "merchant"].edge_index
        src_emb = x_dict["account"][src]
        dst_emb = x_dict["merchant"][dst]
        edge_feat = torch.cat([src_emb, dst_emb], dim=-1)

        logit = self.edge_mlp(edge_feat).squeeze(-1)  # (num_edges,)
        return logit


# -----------------------------
# 4) آماده‌سازی mask های train/val/test
# -----------------------------
def split_edges(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    """
    روی ردیف‌های دیتافریم split می‌زند و خروجی اندیس‌های boolean برای train/val/test است.
    (اینجا تصادفی؛ برای کار واقعی می‌توانی بر اساس زمان split کنی.)
    """
    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(idx, test_size=test_size + val_size, random_state=random_state, stratify=df["label"])
    rel_val = val_size / (test_size + val_size)
    idx_val, idx_test = train_test_split(idx_temp, test_size=1 - rel_val, random_state=random_state, stratify=df["label"].iloc[idx_temp])

    return idx_train, idx_val, idx_test


def build_masks(num_edges, idx_train, idx_val, idx_test):
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    return train_mask, val_mask, test_mask


# -----------------------------
# 5) حلقه آموزش
# -----------------------------
def train_model(data: HeteroData, train_mask, val_mask, epochs=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = HeteroGNN(in_channels=data["account"].x.size(1), hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))  # چون کلاس مثبت کم است

    y = data["account", "transacts", "merchant"].y.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data)
        loss = loss_fn(logits[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds_val = torch.sigmoid(logits[val_mask])
                y_val = y[val_mask]
                val_loss = loss_fn(logits[val_mask], y_val).item()
                try:
                    auc_val = roc_auc_score(y_val.cpu().numpy(), preds_val.cpu().numpy())
                except ValueError:
                    auc_val = np.nan
            print(f"Epoch {epoch:02d} | train loss = {loss.item():.4f} | val loss = {val_loss:.4f} | val AUC = {auc_val:.4f}")

    return model


def evaluate_model(model, data: HeteroData, test_mask):
    device = next(model.parameters()).device
    data = data.to(device)
    y = data["account", "transacts", "merchant"].y.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
        y_true = y[test_mask].cpu().numpy()
        y_pred = (probs >= 0.5).astype(int)

    print("=== Test classification report ===")
    print(classification_report(y_true, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_true, probs)
        print(f"Test AUC-ROC: {auc:.4f}")
    except ValueError:
        print("AUC-ROC قابل محاسبه نیست (شاید همه‌ی برچسب‌ها یکسان بوده‌اند).")


# -----------------------------
# 6) main
# -----------------------------
def main():
    csv_path = "transactions.csv"  # این را به مسیر واقعی فایل خودت عوض کن

    df = load_transactions(csv_path)
    df, num_accounts, num_merchants = encode_ids(df)

    data = build_hetero_graph(df, num_accounts, num_merchants)

    # split
    idx_train, idx_val, idx_test = split_edges(df, test_size=0.2, val_size=0.1)
    num_edges = data["account", "transacts", "merchant"].edge_index.size(1)
    train_mask, val_mask, test_mask = build_masks(num_edges, idx_train, idx_val, idx_test)

    data["account", "transacts", "merchant"].train_mask = train_mask
    data["account", "transacts", "merchant"].val_mask = val_mask
    data["account", "transacts", "merchant"].test_mask = test_mask

    model = train_model(data, train_mask, val_mask, epochs=30, lr=1e-3)
    evaluate_model(model, data, test_mask)


if __name__ == "__main__":
    main()
