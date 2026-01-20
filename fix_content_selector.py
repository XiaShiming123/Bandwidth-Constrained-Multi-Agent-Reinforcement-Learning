#!/usr/bin/env python3
"""
修复ContentSelector维度问题
"""

import re

# 读取文件
with open('experiments/algorithms/adaptive_comm.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找ContentSelector类
pattern = r'class ContentSelector\(nn\.Module\):.*?def forward'

# 替换整个ContentSelector类
new_content_selector = '''class ContentSelector(nn.Module):
    """内容选择器"""
    
    def __init__(self, obs_dim, comm_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.comm_dim = comm_dim
        
        # 重要性网络输出与消息维度相同的权重
        self.importance_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, comm_dim),  # 输出comm_dim
            nn.Sigmoid()  # 输出重要性权重
        )
    
    def forward(self, obs, message):
        # 计算消息各部分的重要性
        importance_weights = self.importance_net(obs)
        
        # 根据重要性选择内容
        selected_content = message * importance_weights
        
        return selected_content'''

# 替换
start = content.find('class ContentSelector(nn.Module):')
if start != -1:
    # 找到下一个类的开始
    next_class = content.find('\nclass ', start + 1)
    if next_class == -1:
        next_class = len(content)
    
    # 替换
    new_content = content[:start] + new_content_selector + content[next_class:]
    
    # 写回文件
    with open('experiments/algorithms/adaptive_comm.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("ContentSelector已修复")
else:
    print("未找到ContentSelector类")

# 验证
with open('experiments/algorithms/adaptive_comm.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'ContentSelector' in line and 'class' in line:
            print(f"找到ContentSelector类在第{i+1}行")
            # 打印接下来的几行
            for j in range(i, min(i+20, len(lines))):
                print(f"  {j+1}: {lines[j].rstrip()}")