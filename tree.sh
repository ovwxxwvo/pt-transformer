#!/usr/bin/env bash


# 项目根目录（脚本所在目录，可手动修改）
PROJ_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GITIGNORE="$PROJ_ROOT/.gitignore"

# 检查 tree 命令是否安装
if ! command -v tree &> /dev/null; then
    echo "❌ 请先安装 tree 命令: sudo apt install tree (Debian/Ubuntu) 或 brew install tree (macOS)"
    exit 1
fi

# 读取 .gitignore 生成排除参数
EXCLUDE_ARGS=()
if [ -f "$GITIGNORE" ]; then
    while IFS= read -r line; do
        # 跳过注释、空行、.git 本身
        [[ -z "$line" || "$line" =~ ^# || "$line" == ".git" ]] && continue
        # 处理目录规则（末尾加 /）
        clean_line=$(echo "$line" | sed -e 's/^[\/]*//' -e 's/[\/]*$//')
        EXCLUDE_ARGS+=("--ignore=$clean_line")
    done < "$GITIGNORE"
else
    echo "⚠️  未找到 .gitignore 文件，将生成完整目录树"
fi

# 生成目录树（-N 显示中文，-F 标记目录/可执行文件）
tree -N "${EXCLUDE_ARGS[@]}" "$PROJ_ROOT"


