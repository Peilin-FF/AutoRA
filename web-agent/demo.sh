#!/bin/bash

echo "==================================================="
echo "多模态RAG文献研究助手演示 (Linux/macOS版)"
echo "==================================================="

# 检查是否存在venv目录
if [ ! -d "venv" ]; then
    echo "正在创建虚拟环境..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "创建虚拟环境失败，请确保已安装Python 3.8或更高版本。"
        exit 1
    fi
fi

# 激活虚拟环境
echo "正在激活虚拟环境..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "激活虚拟环境失败。"
    exit 1
fi

# 安装依赖
echo "正在安装依赖..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "安装依赖失败。"
    exit 1
fi

# 询问用户是否启用讨论功能
read -p "是否启用论文讨论功能？(y/n，默认为n): " enable_discussion
enable_discussion=${enable_discussion:-n}

# 运行演示脚本
echo "正在运行演示..."
if [ "${enable_discussion,,}" = "y" ]; then
    echo "启用论文讨论功能..."
    python run_demo.py --with-discussion
else
    python run_demo.py
fi

if [ $? -ne 0 ]; then
    echo "运行演示失败。"
    exit 1
fi

# 询问用户是否查看JSON结果
read -p "是否查看JSON格式的结果？(y/n，默认为n): " view_results
view_results=${view_results:-n}

if [ "${view_results,,}" = "y" ]; then
    echo "显示结果..."
    python -m json.tool multimodal_rag_analysis.json | more
    
    if [ "${enable_discussion,,}" = "y" ]; then
        echo ""
        read -p "是否查看讨论记录？(y/n，默认为n): " view_discussion
        view_discussion=${view_discussion:-n}
        
        if [ "${view_discussion,,}" = "y" ]; then
            echo "显示讨论记录..."
            ls -1 discussions/*.json 2>/dev/null
            read -p "请输入要查看的讨论文件名(默认为第一个文件): " discussion_file
            
            if [ -z "$discussion_file" ]; then
                discussion_file=$(ls -1 discussions/*.json 2>/dev/null | head -1)
            else
                discussion_file="discussions/$discussion_file"
            fi
            
            if [ -f "$discussion_file" ]; then
                python -m json.tool "$discussion_file" | more
            else
                echo "未找到讨论文件。"
            fi
        fi
    fi
fi

echo ""
echo "==================================================="
echo "演示完成!"
echo "==================================================="

# 询问是否退出
read -p "按回车键退出" exit_choice

# 停用虚拟环境
deactivate 