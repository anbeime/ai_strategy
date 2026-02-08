#!/usr/bin/python
"""
路径更新脚本
将所有文件中的绝对路径 C:/F/ 更新为 C:/F/ai_strategy/
"""

import os
import re
from pathlib import Path

def update_paths_in_file(file_path):
    """更新单个文件中的路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 更新路径模式
        patterns = [
            # C:/F/xxx.py -> C:/F/ai_strategy/xxx.py (但不包括 C:/F/ai_strategy/)
            (r'C:/F/(?!ai_strategy/)([a-zA-Z_][a-zA-Z0-9_]*\.(?:py|txt|json|csv))', r'C:/F/ai_strategy/\1'),
            # C:\F\xxx.py -> C:\F\ai_strategy\xxx.py
            (r'C:\\F\\(?!ai_strategy\\)([a-zA-Z_][a-zA-Z0-9_]*\.(?:py|txt|json|csv))', r'C:\\F\\ai_strategy\\\1'),
            # "C:/F/stock_pool -> "C:/F/ai_strategy/../stock_pool (父目录)
            (r'"C:/F/stock_pool', r'"C:/F/ai_strategy/../stock_pool'),
            (r"'C:/F/stock_pool", r"'C:/F/ai_strategy/../stock_pool"),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, file_path
        
        return False, None
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False, None

def main():
    """主函数"""
    ai_strategy_dir = Path("C:/F/ai_strategy")
    
    if not ai_strategy_dir.exists():
        print(f"错误: 目录不存在 {ai_strategy_dir}")
        return
    
    print("=" * 60)
    print("路径更新脚本")
    print("=" * 60)
    print(f"扫描目录: {ai_strategy_dir}")
    print()
    
    updated_files = []
    
    # 遍历所有Python文件
    for py_file in ai_strategy_dir.glob("*.py"):
        if py_file.name == "update_paths.py":  # 跳过自己
            continue
        
        updated, file_path = update_paths_in_file(py_file)
        if updated:
            updated_files.append(file_path)
            print(f"✓ 已更新: {py_file.name}")
    
    # 更新配置文件
    for config_file in ai_strategy_dir.glob("*.json"):
        updated, file_path = update_paths_in_file(config_file)
        if updated:
            updated_files.append(file_path)
            print(f"✓ 已更新: {config_file.name}")
    
    print()
    print("=" * 60)
    print(f"总计更新了 {len(updated_files)} 个文件")
    print("=" * 60)
    
    if updated_files:
        print("\n更新的文件列表:")
        for f in updated_files:
            print(f"  - {Path(f).name}")
    else:
        print("\n没有文件需要更新")

if __name__ == "__main__":
    main()
