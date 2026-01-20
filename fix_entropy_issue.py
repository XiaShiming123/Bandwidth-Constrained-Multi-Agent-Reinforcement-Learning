#!/usr/bin/env python3
"""
修复熵计算问题
"""

import re

# 读取文件
with open('experiments/algorithms/adaptive_comm.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找并修复熵计算
pattern = r'def _measure_decision_confidence.*?entropy = action_dist\.entropy\(\)\.item\(\)'

# 替换为正确的版本
new_content = re.sub(
    r'entropy = action_dist\.entropy\(\)\.item\(\)',
    'entropy = action_dist.entropy().mean().item()',
    content
)

# 写回文件
with open('experiments/algorithms/adaptive_comm.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("熵计算问题已修复")

# 验证修复
with open('experiments/algorithms/adaptive_comm.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'entropy = action_dist' in line:
            print(f"第{i+1}行: {line.rstrip()}")