#!/usr/bin/env bash
# DNA 基因表达预测 — 端到端运行脚本
#
# 用法:
#   bash run_all.sh              # 完整流程 (v4 全特征)
#   bash run_all.sh --quick      # 快速模式 (仅 v3 基线)
#   bash run_all.sh --step 3     # 从指定步骤开始 (1=数据, 2=v3训练, 3=ENCODE, 4=v4训练, 5=XAI, 6=图表)
#
# 前置条件:
#   conda activate dna-cnn
#   data/ 下已放置 train.h5, valid.h5, test.h5

set -euo pipefail

# 颜色
GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'
CYAN='\033[36m'; BLUE='\033[34m'; GRAY='\033[90m'; RESET='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

quick=false
start_step=1

for arg in "$@"; do
    case $arg in
        --quick)  quick=true ;;
        --step)   shift; start_step=${1:-1} ;;
        --help|-h)
            echo "用法: bash run_all.sh [--quick] [--step N]"
            echo "  --quick    仅运行 v3 基线 (跳过 ENCODE/v4)"
            echo "  --step N   从步骤 N 开始 (1-6)"
            exit 0 ;;
    esac
done

step() {
    local n=$1 name=$2
    echo ""
    echo -e "${BLUE}步骤 ${n}: ${name}${RESET}"
    echo -e "${GRAY}──────────────────────────────────${RESET}"
}

ok()   { echo -e "  ${GREEN}$1${RESET}"; }
warn() { echo -e "  ${YELLOW}$1${RESET}"; }
fail() { echo -e "  ${RED}$1${RESET}"; exit 1; }

# ─── 步骤 1: 校验数据 ───
if [ $start_step -le 1 ]; then
    step 1 "校验数据文件"
    if python script/setup_data.py --check; then
        ok "数据校验通过"
    else
        fail "数据文件缺失，请先运行 python script/setup_data.py"
    fi
fi

# ─── 步骤 2: 训练 v3 基线 ───
if [ $start_step -le 2 ]; then
    step 2 "训练 v3 基线模型 (CNN + Transformer)"
    python script/train_v3.py
    ok "v3 训练完成"
fi

if [ "$quick" = true ]; then
    echo ""
    ok "快速模式完成"
    exit 0
fi

# ─── 步骤 3: ENCODE 表观信号 ───
if [ $start_step -le 3 ]; then
    step 3 "准备 ENCODE 表观信号"
    if [ -f "data/epigenomic.pt" ]; then
        ok "epigenomic.pt 已存在，跳过"
    else
        warn "下载 ENCODE bigWig 文件 (~880MB)，需要网络..."
        python script/prepare_epigenomic.py
        ok "ENCODE 特征提取完成"
    fi
fi

# ─── 步骤 4: 预计算序列特征 + 训练 v4 ───
if [ $start_step -le 4 ]; then
    step 4 "预计算序列特征"
    if [ -f "data/seq_features_train.pt" ]; then
        ok "序列特征已存在，跳过"
    else
        python script/precompute_seq_features.py
        ok "序列特征预计算完成"
    fi

    step "4b" "训练 v4 全特征模型"
    python script/train_v4.py --features all
    ok "v4 训练完成"
fi

# ─── 步骤 5: XAI 可解释性分析 ───
if [ $start_step -le 5 ]; then
    step 5 "可解释性分析 (DeepLIFT + IG)"
    python script/xai_analyze.py
    ok "v3 XAI 分析完成"

    if [ -f "data/modelv4_all_best.pt" ]; then
        python script/xai_analyze_v4.py --features all
        ok "v4 XAI 分析完成"
    else
        warn "v4 模型不存在，跳过 v4 XAI"
    fi
fi

# ─── 步骤 6: 生成论文图表 ───
if [ $start_step -le 6 ]; then
    step 6 "生成论文图表"
    python script/generate_paper_figures.py
    ok "论文图表已生成到 docs/paper_figures/"
fi

echo ""
echo -e "${GREEN}全部流程完成${RESET}"
echo -e "  论文:  ${BLUE}paper/paper.md${RESET}"
echo -e "  图表:  ${BLUE}paper/figures/${RESET}"
echo -e "  XAI:   ${BLUE}results/xai_v4/${RESET}"
echo -e "  日志:  ${BLUE}logs/experiments.csv${RESET}"
