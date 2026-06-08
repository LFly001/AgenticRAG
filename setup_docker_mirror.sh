#!/usr/bin/env bash
# ============================================================
# 配置 Docker 国内镜像加速器
# 解决 docker pull 速度慢 / 超时的问题
# ============================================================
set -euo pipefail

MIRRORS=(
    "https://docker.m.daocloud.io"
    "https://docker.1ms.run"
)

# --- 检测系统，确定 daemon.json 路径 ---
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    DAEMON_FILE="/etc/docker/daemon.json"
    RESTART_CMD="sudo systemctl restart docker"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    DAEMON_FILE="$HOME/.docker/daemon.json"
    RESTART_CMD='echo "请手动重启 Docker Desktop（菜单栏图标 → Restart）"'
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    DAEMON_FILE="$USERPROFILE/.docker/daemon.json"
    RESTART_CMD='echo "请手动重启 Docker Desktop（托盘图标 → Restart）"'
else
    # WSL / 其他
    DAEMON_FILE="/etc/docker/daemon.json"
    RESTART_CMD="sudo systemctl restart docker 2>/dev/null || echo '请手动重启 Docker'"
fi

echo "==> 检测到 daemon.json 路径: $DAEMON_FILE"

# --- 备份现有配置 ---
if [ -f "$DAEMON_FILE" ]; then
    BACKUP="${DAEMON_FILE}.bak.$(date +%s)"
    sudo cp "$DAEMON_FILE" "$BACKUP"
    echo "==> 已备份现有配置到: $BACKUP"
fi

# --- 检查是否已配置过 ---
if [ -f "$DAEMON_FILE" ] && grep -q "registry-mirrors" "$DAEMON_FILE" 2>/dev/null; then
    echo "==> registry-mirrors 已存在，跳过配置。"
    echo "    当前配置："
    grep -A 10 "registry-mirrors" "$DAEMON_FILE"
    exit 0
fi

# --- 写入新配置 ---
MIRRORS_JSON=$(printf '    "%s",\n' "${MIRRORS[@]}")
MIRRORS_JSON=${MIRRORS_JSON%,$'\n'}  # 去掉末尾逗号

if [ -f "$DAEMON_FILE" ]; then
    # 文件已存在但没有 registry-mirrors → 合并
    echo "==> 在现有 daemon.json 中添加 registry-mirrors..."
    # 用 Python 合并 JSON（兼容性好）
    python3 -c "
import json, sys
with open('$DAEMON_FILE') as f:
    cfg = json.load(f)
cfg['registry-mirrors'] = ${MIRRORS[@]@Q}  # 这里用 python 直接写
" 2>/dev/null || {
        # Python 不可用时，直接覆盖（已备份）
        echo "==> Python 不可用，覆盖写入..."
        sudo mkdir -p "$(dirname "$DAEMON_FILE")"
        sudo tee "$DAEMON_FILE" > /dev/null <<EOF
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://docker.1ms.run"
    ]
}
EOF
    }
else
    # 文件不存在 → 新建
    echo "==> 创建 daemon.json..."
    sudo mkdir -p "$(dirname "$DAEMON_FILE")"
    sudo tee "$DAEMON_FILE" > /dev/null <<EOF
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://docker.1ms.run"
    ]
}
EOF
fi

echo "==> 配置写入完成！"
echo ""
echo "   下一步：重启 Docker 使配置生效"
echo "   $RESTART_CMD"
echo ""

# --- 尝试验证 ---
echo "==> 尝试验证镜像源连通性..."
for mirror in "${MIRRORS[@]}"; do
    if curl -s --connect-timeout 3 "$mirror/v2/" > /dev/null 2>&1; then
        echo "   ✓ $mirror — 可达"
    else
        echo "   ✗ $mirror — 不可达"
    fi
done
