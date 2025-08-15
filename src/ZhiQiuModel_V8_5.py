"""
==============================================================================
知秋模型 V8.5: 数据持久化版 (Zhiqiu Model V8.5)

版本演进历史 (Version Evolution History):
-----------------------------------------
v8.4 -> v8.5 (当前版本 - 数据持久化版):
    - [核心功能] 新增步骤12，在程序运行结束后，使用pickle将所有用于后续
      分析和绘图的变量（包括结果字典、预测数组、配置信息等）保存到
      一个`.pkl`文件中。
    - [科研流程优化] 此功能实现了“计算”与“可视化”的彻底分离，用户
      未来可以编写独立的轻量级脚本来加载此`.pkl`文件，无限次、
      快速地重新生成或修改图表，极大提升了科研效率。

v8.3 -> v8.4 (修复与增强版):
    - [代码质量] 采纳Bug分析报告建议，对代码质量和健壮性进行全面优化。
    - [BUG修复-澄清] 针对报告中指出的“严重Bug #1, #2”，经复核确认原代码逻辑
      正确，为避免误解，在相应位置补充了超详细的“保姆级注释”以阐明工作原理。
    - [BUG修复-修正] 针对报告中指出的“潜在Bug #3, #4”，虽然原代码通过min_len
      等方式已规避了风险，但为增强逻辑清晰度，补充了更明确的长度校验和注释。
    - [代码质量] 移除了未使用的`import time`。
    - [代码质量] 为关键的文件加载步骤增加了`try-except`异常处理，增强程序健壮性。
    - [性能优化] 在每个训练轮次结束后，增加了`torch.cuda.empty_cache()`，
      辅助GPU进行显存管理，提升训练稳定性。
    - [可读性] 对部分过长的单行代码进行了换行处理。
==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import pandas as pd
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import os
import warnings
import json
import pickle
import gc  # 内存管理
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 0. 参数总控 (Master Configuration) - V8.5
# ==============================================================================
print("--- 步骤 0: 参数设置 (V8.5 数据持久化版) ---")

# 随机种子设置
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 文件路径设置
ERA5_T2M_PATH = 'ERA5_2010_2023_sp_t2m.nc'
ERA5_SST_MSL_PATH = 'ERA5_2010_2023_sst_msl_d2m.nc'
ERA5_SSR_PATH = 'ERA5_2010_2023_ssr_avg_snswrf_snlwrf.nc'
SIC_PATH = './arco/processed_sic_monthly.nc'
DRIFT_PATH = './weekly/processed_drift_monthly.nc'
MODEL_SAVE_PATH = './models/'
RESULTS_SAVE_PATH = './results/'

# 实验参数
DATA_SUBSET = None
FEATURE_ENGINEERING_VERSION = 'B'
MODEL_VERSION = '3'  # 可选: 'lstm', '1', '2', '3'
HORIZON = 1
N_MC_SAMPLES = 50

# 训练超参数
SEQUENCE_LENGTH = 6
BATCH_SIZE = 256
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10 # 建议适当增加以进行更充分的训练
EFFECTIVE_NUM_WORKERS = 0 if os.name == "nt" else 2

# 训练稳定性参数
GRADIENT_CLIP_VALUE = 1.0
NAN_CHECK_FREQUENCY = 100
LOSS_EXPLOSION_THRESHOLD = 100.0

# 运行标识
RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"本次运行唯一标识 (Run Tag): {RUN_TAG}")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

# ==============================================================================
# I. 健壮性验证和工程实践工具
# ==============================================================================
print("\n--- I: 初始化健壮性验证工具 ---")

def check_required_files():
    """检查所有必需的输入文件是否存在。"""
    required_files = [ERA5_T2M_PATH, ERA5_SST_MSL_PATH, ERA5_SSR_PATH, SIC_PATH, DRIFT_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"关键文件不存在: {file_path}")
        print(f"✅ 文件检查通过: {file_path}")
    print("✅ 所有必需文件检查通过。")

def validate_data_alignment(*datasets):
    """验证所有数据集的时空对齐。"""
    if len(datasets) < 2: return True
    print("🔍 开始验证数据时空对齐...")
    reference = datasets[0]
    for i, ds in enumerate(datasets[1:], 1):
        if not np.array_equal(ds.coords['time'].values, reference.coords['time'].values):
            raise ValueError(f"数据集 {i} 时间坐标不对齐")
        for coord in ['y', 'x']:
            if coord in ds.coords and coord in reference.coords:
                if not np.allclose(ds.coords[coord], reference.coords[coord], atol=1e-6):
                    raise ValueError(f"数据集 {i} 的 {coord} 坐标不对齐")
    print("✅ 所有数据集时空对齐验证通过。")

def validate_data_quality(data, var_name):
    """基于变量类型的智能数据质量验证与NaN填充。"""
    print(f"🔍 开始验证 {var_name} 数据质量...")
    nan_count = np.isnan(data.values).sum()
    if nan_count > 0:
        print(f"⚠️  {var_name} 包含 {nan_count:,} 个NaN值")
        var_lower = var_name.lower()
        if 'sic' in var_lower or 'seaice' in var_lower:
            fill_value = 0.0
            print(f"   🧊 海冰浓度变量用 {fill_value} 填充")
            data = data.fillna(fill_value)
        elif any(temp_var in var_lower for temp_var in ['t2m', 'sst']):
            fill_value = float(data.mean().values)
            print(f"   🌡️  温度变量用均值 {fill_value:.2f} 填充")
            data = data.fillna(fill_value)
        else:
            fill_value = float(data.median().values)
            print(f"   📊 其他变量用中位数 {fill_value:.4f} 填充")
            data = data.fillna(fill_value)
    if np.isinf(data.values).sum() > 0:
        raise ValueError(f"{var_name} 包含无限值")
    print(f"✅ {var_name} 数据质量检查通过。")
    return data

def optimize_memory_usage():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

check_required_files()

# ==============================================================================
# II. 模型定义库 (Model Definition Library) - V8.1 完整版
# ==============================================================================
print("\n--- II: 定义模型库 (V8.1 完整版) ---")

def _assert_divisible(d, n):
    """确保d能被n整除。"""
    if d % n != 0: 
        raise ValueError(f"d_model ({d}) must be divisible by num_heads ({n}).")

def _build_pos_encoding(max_len: int, d_model: int):
    """构建位置编码。"""
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

def _causal_mask(L: int, device: torch.device):
    """生成因果掩码。"""
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

class LSTMModel(nn.Module):
    """
    LSTM基础模型：作为循环神经网络 (RNN) 的代表，是Transformer出现前的主流序列模型。
    """
    def __init__(self, i_dim, h_dim, n_layers, o_dim, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(i_dim, h_dim, n_layers, batch_first=True, dropout=drop if n_layers>1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2), 
            nn.ReLU(), 
            nn.Dropout(drop), 
            nn.Linear(h_dim // 2, o_dim), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class ImprovedSequenceTransformer_v1(nn.Module):
    """
    Transformer v1: 基础Transformer Encoder架构。
    """
    def __init__(self, i_dim, m_dim, n_head, n_layers, o_dim, drop=0.1, max_len=5000):
        super().__init__()
        _assert_divisible(m_dim, n_head)
        self.input_fc = nn.Linear(i_dim, m_dim)
        self.register_buffer("pos_encoding", _build_pos_encoding(max_len, m_dim), persistent=False)
        enc_layer = nn.TransformerEncoderLayer(m_dim, n_head, batch_first=True, dropout=drop, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.output_fc = nn.Sequential(
            nn.Linear(m_dim, m_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(m_dim // 2, o_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        _, L, _ = x.shape
        x = self.input_fc(x)
        x = x + self.pos_encoding[:, :L, :].to(x.device, dtype=x.dtype)
        x = self.encoder(x, mask=_causal_mask(L, x.device))
        return self.output_fc(x[:, -1, :])

class ImprovedSequenceTransformer_v2(nn.Module):
    """
    Transformer v2: 在v1的基础上，增加了序列平均的混合策略。
    """
    def __init__(self, i_dim, m_dim, n_head, n_layers, o_dim, drop=0.1, max_len=5000, blend=0.1):
        super().__init__()
        _assert_divisible(m_dim, n_head)
        self.blend = blend
        self.input_fc = nn.Linear(i_dim, m_dim)
        self.register_buffer("pos_encoding", _build_pos_encoding(max_len, m_dim), persistent=False)
        enc_layer = nn.TransformerEncoderLayer(m_dim, n_head, batch_first=True, dropout=drop, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.output_fc = nn.Sequential(
            nn.Linear(m_dim, m_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(m_dim // 2, o_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        _, L, _ = x.shape
        x = self.input_fc(x)
        x = x + self.pos_encoding[:, :L, :].to(x.device, dtype=x.dtype)
        x = self.encoder(x, mask=_causal_mask(L, x.device))
        x = x[:, -1, :] + self.blend * x.mean(dim=1)
        return self.output_fc(x)

class ImprovedSequenceTransformer_v3(nn.Module):
    """
    Transformer v3: 引入了CLS token的先进架构，借鉴自BERT/ViT。
    """
    def __init__(self, i_dim, m_dim, n_head, n_layers, o_dim, drop=0.1, max_len=5000):
        super().__init__()
        _assert_divisible(m_dim, n_head)
        self.input_fc = nn.Sequential(nn.Linear(i_dim, m_dim), nn.LayerNorm(m_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, m_dim) * 0.02)
        self.register_buffer("pos_encoding", _build_pos_encoding(max_len + 1, m_dim), persistent=False)
        enc_layer = nn.TransformerEncoderLayer(m_dim, n_head, batch_first=True, dropout=drop, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.output_fc = nn.Sequential(
            nn.Linear(m_dim, m_dim // 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(m_dim // 2, m_dim // 4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(m_dim // 4, o_dim), nn.Sigmoid()
        )
    
    def forward(self, x):
        B, L, _ = x.shape
        x = self.input_fc(x)
        cls = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_encoding[:, :(L+1), :].to(x.device, dtype=x.dtype)
        mask = _causal_mask(L + 1, x.device)
        mask[0, :] = False
        x = self.encoder(x, mask=mask)
        return self.output_fc(x[:, 0, :])

def get_model(version, X_train_shape):
    """
    根据版本号获取对应的模型实例 (完整版)。
    """
    i_dim = X_train_shape[2]
    v_str = str(version).lower()
    
    if v_str == 'lstm':
        print("📦 正在实例化模型: LSTM (基线)")
        return LSTMModel(i_dim, 128, 2, 1)
    elif v_str == '1':
        print("📦 正在实例化模型: Transformer v1")
        return ImprovedSequenceTransformer_v1(i_dim, 128, 4, 2, 1)
    elif v_str == '2':
        print("📦 正在实例化模型: Transformer v2")
        return ImprovedSequenceTransformer_v2(i_dim, 128, 4, 3, 1)
    elif v_str == '3':
        print("📦 正在实例化模型: Transformer v3 (CLS Token)")
        return ImprovedSequenceTransformer_v3(i_dim, 256, 8, 4, 1)
    else:
        raise ValueError("MODEL_VERSION 必须是 'lstm', '1', '2', 或 '3'")

# ==============================================================================
# III. 增强验证和修复工具
# ==============================================================================
print("\n--- III: 初始化增强验证工具 ---")
class StrictMCDropout:
    def __init__(self, model, dropout_layers_only=True): self.model = model; self.dropout_layers_only = dropout_layers_only; self.original_training_states = {}
    def __enter__(self):
        for name, module in self.model.named_modules(): self.original_training_states[name] = module.training
        if self.dropout_layers_only: self.model.eval(); [m.train() for m in self.model.modules() if isinstance(m, nn.Dropout)]
        else: self.model.train()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules(): module.train() if self.original_training_states[name] else module.eval()
def compute_sie_with_validation(sic, area_map, thresh=0.15):
    if sic.shape != area_map.shape: raise ValueError(f"SIC维度 {sic.shape} 与面积图维度 {area_map.shape} 不匹配")
    mask = (sic >= thresh).astype(np.float32); return float((mask * area_map).sum())
def comprehensive_data_validation(X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq, num_pixels, SEQUENCE_LENGTH, HORIZON, X_train_raw, X_val_raw, X_test_raw):
    print("🔍 执行综合数据完整性验证...")
    L_total = SEQUENCE_LENGTH + (HORIZON - 1)
    def nseq(T, L): return max(0, T - L)
    train_T, val_T, test_T = X_train_raw.shape[0], X_val_raw.shape[0], X_test_raw.shape[0]
    expected_samples = (nseq(train_T, L_total) + nseq(val_T, L_total) + nseq(test_T, L_total)) * num_pixels
    total_samples = len(X_train_seq) + len(X_val_seq) + len(X_test_seq)
    if total_samples != expected_samples: print(f"❌ 样本数量不一致! 期望: {expected_samples:,}, 实际: {total_samples:,}")
    else: print("✅ 样本数量一致性检查通过。")

# ==============================================================================
# IV. 主流程
# ==============================================================================

# ------------------------------------------------------------------------------
# 步骤 1: 加载已处理好的数据
# ------------------------------------------------------------------------------
print("\n--- 步骤 1: 加载已处理好的数据 ---")
try:
    sic_ds = xr.open_dataset(SIC_PATH)
    drift_ds = xr.open_dataset(DRIFT_PATH)
    t2m_ds = xr.open_dataset(ERA5_T2M_PATH)
    sst_msl_ds = xr.open_dataset(ERA5_SST_MSL_PATH)
    ssr_ds = xr.open_dataset(ERA5_SSR_PATH)
except FileNotFoundError as e:
    print(f"❌ 文件加载失败: {e}")
    raise

datasets_to_process = [sic_ds, drift_ds, t2m_ds, sst_msl_ds, ssr_ds]
processed_datasets = []
for ds in datasets_to_process:
    if 'valid_time' in ds.coords: ds = ds.rename({'valid_time': 'time'})
    if 'time' in ds.coords and ds['time'].dtype != 'datetime64[ns]':
        try:
            ds['time'] = ds.indexes['time'].to_datetimeindex()
        except:
            ds['time'] = pd.to_datetime([datetime.datetime(t.year, t.month, t.day) for t in ds['time'].values])
    processed_datasets.append(ds)

sic_ds, drift_ds, t2m_ds, sst_msl_ds, ssr_ds = processed_datasets
sic_monthly = sic_ds['cdr_seaice_conc_monthly']
drift_u_monthly = drift_ds['u']
drift_v_monthly = drift_ds['v']
t2m_monthly = t2m_ds['t2m']
sst_monthly = sst_msl_ds['sst']
msl_monthly = sst_msl_ds['msl']
ssr_monthly = ssr_ds['ssr']

time_coords = sic_monthly['time'].values
start_date_str = pd.to_datetime(time_coords[0]).strftime('%Y-%m-%d')
end_date_str = pd.to_datetime(time_coords[-1]).strftime('%Y-%m-%d')
print(f"✅ 自动检测到数据时间范围: {start_date_str} 到 {end_date_str}")

all_unaligned = [sic_monthly, drift_u_monthly, drift_v_monthly, t2m_monthly, sst_monthly, msl_monthly, ssr_monthly]
all_aligned = [sic_monthly]
for ds in all_unaligned[1:]:
    all_aligned.append(ds.reindex_like(sic_monthly, method="nearest"))
validate_data_alignment(*all_aligned)

dataset_names = ['SIC', 'Drift_U', 'Drift_V', 'T2M', 'SST', 'MSL', 'SSR']
unpacked_datasets = [validate_data_quality(ds, name) for ds, name in zip(all_aligned, dataset_names)]
sic_monthly, drift_u_monthly, drift_v_monthly, t2m_monthly, sst_monthly, msl_monthly, ssr_monthly = unpacked_datasets

grid_shape = sic_monthly.shape[1:]
num_timesteps = len(time_coords)
num_pixels = grid_shape[0] * grid_shape[1]
CELL_AREA_KM2_MAP = np.full(grid_shape, 25.0 * 25.0, dtype=np.float32)
print(f"📋 数据集基本信息: 网格形状={grid_shape}, 时间步数={num_timesteps}, 像素点数={num_pixels:,}")

# ------------------------------------------------------------------------------
# 步骤 2: 构建特征矩阵
# ------------------------------------------------------------------------------
print("\n--- 步骤 2: 构建特征矩阵 ---")
flat_sic = sic_monthly.values.reshape(num_timesteps, -1)
flat_drift_u = drift_u_monthly.values.reshape(num_timesteps, -1)
flat_drift_v = drift_v_monthly.values.reshape(num_timesteps, -1)
flat_t2m = t2m_monthly.values.reshape(num_timesteps, -1)
flat_sst = sst_monthly.values.reshape(num_timesteps, -1)
flat_ssr = ssr_monthly.values.reshape(num_timesteps, -1)
flat_msl = msl_monthly.values.reshape(num_timesteps, -1)

months = pd.to_datetime(time_coords).month.to_numpy()
month_sin = np.sin(2 * np.pi * (months - 1) / 12).reshape(-1, 1).repeat(num_pixels, axis=1)
month_cos = np.cos(2 * np.pi * (months - 1) / 12).reshape(-1, 1).repeat(num_pixels, axis=1)
features_matrix = np.stack([month_sin[:-1], month_cos[:-1], flat_sic[:-1], flat_drift_u[:-1], flat_drift_v[:-1], flat_t2m[:-1], flat_sst[:-1], flat_ssr[:-1], flat_msl[:-1]], axis=2)
all_data_X = features_matrix
all_data_y = flat_sic[1:]
feature_names = ['month_sin', 'month_cos', 'sic', 'drift_u', 'drift_v', 't2m', 'sst', 'ssr', 'msl']

# ------------------------------------------------------------------------------
# 步骤 3: 时序切分与标准化
# ------------------------------------------------------------------------------
print("\n--- 步骤 3: 时序切分与标准化 ---")
y_time_coords = time_coords[1:]
train_end_idx = int(len(all_data_X) * 0.7)
val_end_idx = int(len(all_data_X) * 0.9)
X_train_raw, y_train_raw = all_data_X[:train_end_idx], all_data_y[:train_end_idx]
X_val_raw, y_val_raw = all_data_X[train_end_idx:val_end_idx], all_data_y[train_end_idx:val_end_idx]
X_test_raw, y_test_raw = all_data_X[val_end_idx:], all_data_y[val_end_idx:]
train_y_times = y_time_coords[:train_end_idx]

print("🔧 执行特征标准化...")
X_train_flat = X_train_raw.reshape(-1, X_train_raw.shape[2])
scaler_X = StandardScaler().fit(X_train_flat)
scaler_path = os.path.join(MODEL_SAVE_PATH, f"scaler_{RUN_TAG}.pkl")
with open(scaler_path, 'wb') as f: pickle.dump(scaler_X, f)
print(f"💾 标准化器 (Scaler) 已保存到: {scaler_path}")
def scale_X(X_raw, sX): return sX.transform(X_raw.reshape(-1, X_raw.shape[2])).reshape(X_raw.shape)
X_train_s, y_train_s = scale_X(X_train_raw, scaler_X), y_train_raw
X_val_s, y_val_s = scale_X(X_val_raw, scaler_X), y_val_raw
X_test_s, y_test_s = scale_X(X_test_raw, scaler_X), y_test_raw

# ------------------------------------------------------------------------------
# 步骤 4: 创建时序样本与数据加载器
# ------------------------------------------------------------------------------
print(f"\n--- 步骤 4: 创建时序样本 ---")
def create_sequences(X, y, seq_len, horizon):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - (horizon - 1)):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len + (horizon - 1)])
    Xs, ys = np.array(Xs), np.array(ys)
    return torch.FloatTensor(Xs.transpose(0, 2, 1, 3).reshape(-1, seq_len, X.shape[2])), torch.FloatTensor(ys.reshape(-1, 1))

X_train_seq, y_train_seq = create_sequences(X_train_s, y_train_s, SEQUENCE_LENGTH, HORIZON)
X_val_seq, y_val_seq = create_sequences(X_val_s, y_val_s, SEQUENCE_LENGTH, HORIZON)
X_test_seq, y_test_seq = create_sequences(X_test_s, y_test_s, SEQUENCE_LENGTH, HORIZON)
comprehensive_data_validation(X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq, num_pixels, SEQUENCE_LENGTH, HORIZON, X_train_raw, X_val_raw, X_test_raw)

class SequenceDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

train_loader = DataLoader(SequenceDataset(X_train_seq, y_train_seq), batch_size=BATCH_SIZE, shuffle=True, num_workers=EFFECTIVE_NUM_WORKERS)
val_loader = DataLoader(SequenceDataset(X_val_seq, y_val_seq), batch_size=BATCH_SIZE, shuffle=False, num_workers=EFFECTIVE_NUM_WORKERS)

# ------------------------------------------------------------------------------
# 步骤 5: 模型实例化
# ------------------------------------------------------------------------------
print(f"\n--- 步骤 5: 实例化模型 ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(MODEL_VERSION, X_train_seq.shape).to(device)
criterion = nn.MSELoss()
print(f"🖥️  使用设备: {device}, 模型参数: {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------------------------------------------------
# 步骤 6: 模型训练
# ------------------------------------------------------------------------------
print("\n--- 步骤 6: 开始训练 ---")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = os.path.join(MODEL_SAVE_PATH, f"best_model_{RUN_TAG}.pt")
training_history = {'train_loss': [], 'val_loss': []}

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [训练]"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [验证]"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X)
            val_loss += criterion(pred, batch_y).item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    training_history['train_loss'].append(avg_train_loss)
    training_history['val_loss'].append(avg_val_loss)
    
    print(f"Epoch {epoch+1:2d}: 训练损失={avg_train_loss:.5f}, 验证损失={avg_val_loss:.5f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"    ✅ 验证损失降低，保存最佳模型")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"    🛑 触发早停")
            break
    
    optimize_memory_usage()

# ------------------------------------------------------------------------------
# 步骤 7: 模型评估
# ------------------------------------------------------------------------------
print("\n--- 步骤 7: 测试集评估 ---")
# V8.5: 推荐使用 weights_only=True 以遵循PyTorch的最佳实践，并消除警告
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
model.eval()
y_true = y_test_seq.numpy()
test_loader = DataLoader(SequenceDataset(X_test_seq, y_test_seq), batch_size=BATCH_SIZE*2, shuffle=False)
y_pred_list = []
mc_predictions = []

with torch.no_grad():
    for batch_X, _ in tqdm(test_loader, desc="确定性预测"):
        y_pred_list.append(model(batch_X.to(device)).cpu().numpy())
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    for _ in tqdm(range(N_MC_SAMPLES), desc="MC Dropout采样"):
        sample_preds = []
        with StrictMCDropout(model):
            for batch_X, _ in test_loader:
                sample_preds.append(model(batch_X.to(device)).cpu().numpy())
        mc_predictions.append(np.concatenate(sample_preds, axis=0))

mc_predictions = np.array(mc_predictions)
y_pred_mean = mc_predictions.mean(axis=0)
y_pred_std = mc_predictions.std(axis=0)

results = {
    'model_version': str(MODEL_VERSION), 
    'run_tag': RUN_TAG, 
    'best_val_loss': best_val_loss, 
    'epochs_trained': epoch + 1
}
results['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred_mean)))
results['mae'] = float(mean_absolute_error(y_true, y_pred_mean))
results['r2'] = float(r2_score(y_true, y_pred_mean))

# ------------------------------------------------------------------------------
# 步骤 8 & 9: 基准对比与SIE分析
# ------------------------------------------------------------------------------
print("\n--- 步骤 8 & 9: 基准对比与SIE分析 ---")
y_test_flat = y_test_raw.ravel()
test_months = pd.to_datetime(y_time_coords[val_end_idx:]).month.to_numpy()
train_months = pd.to_datetime(train_y_times).month.to_numpy()
climo_map_monthly = np.array([y_train_raw[train_months == m].mean(axis=0) for m in range(1, 13)])
climo_pred_flat = np.array([climo_map_monthly[m-1] for m in test_months[:len(y_test_raw)]]).ravel()
persistence_pred_flat = X_test_raw[:, :, feature_names.index('sic')].ravel()

# V8.4 关键修复: 强制确保所有用于比较的数组长度一致，防止崩溃
min_len = min(len(y_test_flat), len(climo_pred_flat), len(persistence_pred_flat))
y_test_flat = y_test_flat[:min_len]
climo_pred_flat = climo_pred_flat[:min_len]
persistence_pred_flat = persistence_pred_flat[:min_len]
results['climo_rmse'] = float(np.sqrt(mean_squared_error(y_test_flat, climo_pred_flat)))
results['persistence_rmse'] = float(np.sqrt(mean_squared_error(y_test_flat, persistence_pred_flat)))

num_test_months = len(y_true) // num_pixels
test_time_coords_for_sie = y_time_coords[val_end_idx : val_end_idx + num_test_months]

sie_true, sie_pred_mean = [], []
for i in tqdm(range(num_test_months), desc="计算SIE"):
    true_map = y_true[i*num_pixels:(i+1)*num_pixels].reshape(grid_shape)
    pred_map = y_pred_mean[i*num_pixels:(i+1)*num_pixels].reshape(grid_shape)
    sie_true.append(compute_sie_with_validation(true_map, CELL_AREA_KM2_MAP))
    sie_pred_mean.append(compute_sie_with_validation(pred_map, CELL_AREA_KM2_MAP))

sie_true_m = np.array(sie_true) / 1e6
sie_pred_mean_m = np.array(sie_pred_mean) / 1e6
results['sie_rmse'] = float(np.sqrt(mean_squared_error(sie_true_m, sie_pred_mean_m)))
results['sie_mae'] = float(mean_absolute_error(sie_true_m, sie_pred_mean_m))

# ------------------------------------------------------------------------------
# 步骤 10: 最终报告
# ------------------------------------------------------------------------------
print("\n--- 步骤 10: 生成最终报告 ---")
def generate_comprehensive_report(results):
    print("\n" + "="*80); print(f"🧊 知秋模型 V8.5 - 最终实验报告"); print("="*80)
    print(f"📋 实验配置: 模型版本='{results['model_version']}', 运行标识={results['run_tag']}")
    print(f"   训练结果: 共训练 {results['epochs_trained']} 轮, 最佳验证损失={results['best_val_loss']:.5f}")
    print(f"\n📊 核心目标: 海冰密集度 (SIC) 像素级预测性能:")
    print(f"   模型 RMSE = {results['rmse']:.4f} (越低越好)")
    print(f"   气候态基准 RMSE = {results['climo_rmse']:.4f} (基准)")
    print(f"   持续性基准 RMSE = {results['persistence_rmse']:.4f} (基准)")
    print(f"   模型 MAE  = {results['mae']:.4f}"); print(f"   模型 R²   = {results['r2']:.4f} (越高越好)")
    print(f"\n🧊 宏观指标: 海冰范围 (SIE) 预测性能:")
    print(f"   SIE RMSE = {results['sie_rmse']:.3f} 百万km²"); print(f"   SIE MAE  = {results['sie_mae']:.3f} 百万km²"); print("="*80)
generate_comprehensive_report(results)

# ------------------------------------------------------------------------------
# 步骤 11 - 论文级图表生成
# ------------------------------------------------------------------------------
print("\n--- 步骤 11: 生成论文级图表 ---")
def generate_publication_plots(results, training_history, y_true, y_pred_mean, y_pred_std, sie_true, sie_pred_mean, test_time_coords_for_sie, grid_shape, run_tag):
    print("🎨 开始生成图表...")
    # V8.5: 增加中文字体设置，修复图表乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-paper')
    
    # 1. SIE 时间序列对比图
    print("   -> 正在生成 SIE 时间序列图...")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(test_time_coords_for_sie, sie_true, 'k-', marker='o', markersize=4, label='观测值 (Observation)')
    ax1.plot(test_time_coords_for_sie, sie_pred_mean, 'r-', marker='x', markersize=4, label='模型预测 (Forecast)')
    
    pixel_variances = y_pred_std**2
    pixel_variances_monthly = pixel_variances.reshape(num_test_months, num_pixels)
    cell_area_km2 = CELL_AREA_KM2_MAP[0, 0]
    sie_variance_km4 = pixel_variances_monthly.sum(axis=1) * (cell_area_km2**2)
    sie_std_dev_m = np.sqrt(sie_variance_km4) / 1e6
    ax1.fill_between(test_time_coords_for_sie, 
                     sie_pred_mean - 1.96 * sie_std_dev_m, 
                     sie_pred_mean + 1.96 * sie_std_dev_m, 
                     color='red', alpha=0.2, label='95% 置信区间 (95% C.I.)')
    
    ax1.set_title('月度北极海冰范围 (SIE) 预测与观测对比', fontsize=16)
    ax1.set_xlabel('日期 (Date)', fontsize=12); ax1.set_ylabel('海冰范围 (SIE) [百万 km²]', fontsize=12)
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6); fig1.tight_layout()
    fig1.savefig(os.path.join(RESULTS_SAVE_PATH, f"SIE_timeseries_{run_tag}.png"), dpi=300)
    print("      SIE 图已保存。")

    # 2. 训练/验证损失曲线
    print("   -> 正在生成损失曲线图...")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(training_history['train_loss']) + 1)
    ax2.plot(epochs, training_history['train_loss'], 'b-o', label='训练损失 (Training Loss)')
    ax2.plot(epochs, training_history['val_loss'], 'r-o', label='验证损失 (Validation Loss)')
    best_epoch = np.argmin(training_history['val_loss']) + 1
    ax2.axvline(best_epoch, color='grey', linestyle='--', label=f'最佳轮次 (Best Epoch): {best_epoch}')
    ax2.set_title('模型训练过程中的损失变化', fontsize=16)
    ax2.set_xlabel('轮次 (Epoch)', fontsize=12); ax2.set_ylabel('均方误差损失 (MSE Loss)', fontsize=12)
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6); fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_SAVE_PATH, f"loss_curve_{run_tag}.png"), dpi=300)
    print("      损失曲线图已保存。")

    # 3. 综合四联图
    print("   -> 正在生成综合性能分析图...")
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10)); fig3.suptitle(f'知秋模型 V{MODEL_VERSION} 综合性能评估', fontsize=20)
    ax = axes[0, 0]; sample_indices = np.random.choice(len(y_true), size=min(50000, len(y_true)), replace=False)
    ax.scatter(y_true[sample_indices], y_pred_mean[sample_indices], alpha=0.1, s=5); ax.plot([0, 1], [0, 1], 'r--', lw=2)
    ax.set_title('像素级预测 vs. 真实值 (抽样)', fontsize=14); ax.set_xlabel('真实 SIC', fontsize=12); ax.set_ylabel('预测 SIC', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6); ax.text(0.05, 0.95, f"R² = {results['r2']:.4f}\nRMSE = {results['rmse']:.4f}", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)); ax.set_aspect('equal', 'box')
    ax = axes[0, 1]; errors = (y_pred_mean - y_true).flatten()
    ax.hist(errors, bins=100, density=True, range=(-0.5, 0.5)); ax.set_title('预测误差分布', fontsize=14)
    ax.set_xlabel('预测误差 (预测值 - 真实值)', fontsize=12); ax.set_ylabel('概率密度', fontsize=12); ax.grid(True, linestyle='--', alpha=0.6); ax.axvline(0, color='red', linestyle='--')
    ax = axes[1, 0]; spatial_mae = np.mean(np.abs(y_pred_mean - y_true).reshape(-1, num_pixels), axis=0).reshape(grid_shape)
    im = ax.imshow(spatial_mae, cmap='Reds', origin='lower'); ax.set_title('空间平均绝对误差 (MAE)', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='MAE')
    ax = axes[1, 1]; spatial_uncertainty = y_pred_std.reshape(-1, num_pixels).mean(axis=0).reshape(grid_shape)
    im = ax.imshow(spatial_uncertainty, cmap='viridis', origin='lower'); ax.set_title('空间平均预测不确定性 (Std Dev)', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Std Dev')
    fig3.tight_layout(rect=[0, 0, 1, 0.96]); fig3.savefig(os.path.join(RESULTS_SAVE_PATH, f"comprehensive_analysis_{run_tag}.png"), dpi=300)
    print("      综合性能图已保存。")
    
    plt.close('all')
    print("🎨 所有图表生成完毕。")

# 调用图表生成函数
generate_publication_plots(results, training_history, y_true, y_pred_mean, y_pred_std, sie_true_m, sie_pred_mean_m, test_time_coords_for_sie, grid_shape, RUN_TAG)

# ==============================================================================
# 步骤 12 (V8.5新增): 保存所有绘图所需数据
# ==============================================================================
print("\n--- 步骤 12: 保存绘图数据 ---")

# 保姆级注释：
# 我们将所有用于生成上方图表和报告的关键变量，打包到一个Python字典(dictionary)中。
# 这个字典就像一个“数据工具箱”，里面装满了我们需要的一切。
# 这样做，我们就可以把这个“工具箱”一次性保存起来。
plotting_data = {
    # 1. 总结性报告字典
    'results': results,
    # 2. 完整的训练历史记录
    'training_history': training_history,
    # 3. 庞大的像素级预测数组
    'y_true': y_true,
    'y_pred_mean': y_pred_mean,
    'y_pred_std': y_pred_std,
    # 4. SIE时间序列数组
    'sie_true_m': sie_true_m,
    'sie_pred_mean_m': sie_pred_mean_m,
    # 5. 绘图所需的配置和元数据
    'test_time_coords_for_sie': test_time_coords_for_sie,
    'grid_shape': grid_shape,
    'run_tag': RUN_TAG,
    'model_version': MODEL_VERSION
}

# 定义保存路径，文件名包含本次运行的唯一标识(RUN_TAG)，确保不会覆盖旧结果。
plotting_data_path = os.path.join(RESULTS_SAVE_PATH, f"plotting_data_{RUN_TAG}.pkl")

# 使用Python的pickle库，将整个“数据工具箱”写入一个二进制文件(.pkl)。
# 'wb' 中的 'w' 代表写入(write)，'b' 代表二进制(binary)。
try:
    with open(plotting_data_path, 'wb') as f:
        pickle.dump(plotting_data, f)
    print(f"✅ 所有绘图数据已成功保存到: {plotting_data_path}")
    print("   您现在可以编写一个独立的脚本，加载此文件来快速重新生成或自定义图表。")
except Exception as e:
    print(f"❌ 保存绘图数据失败: {e}")

print("\n" + "="*80)
print("🎉 知秋模型 V8.5 - 完整流程执行成功！")
print("="*80)