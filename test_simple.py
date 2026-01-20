#!/usr/bin/env python3
"""
简化测试脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic():
    """基础测试"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print(f"OK - numpy {np.__version__}")
    except ImportError as e:
        print(f"FAIL - numpy: {e}")
        return False
    
    try:
        import torch
        print(f"OK - torch {torch.__version__}")
    except ImportError as e:
        print(f"FAIL - torch: {e}")
        return False
    
    try:
        import matplotlib
        print(f"OK - matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"FAIL - matplotlib: {e}")
        return False
    
    return True

def test_project_structure():
    """测试项目结构"""
    print("\nTesting project structure...")
    
    required = [
        "experiments/main_experiment.py",
        "experiments/configs/default.yaml",
        "requirements.txt",
        "README.md"
    ]
    
    all_ok = True
    for item in required:
        if os.path.exists(item):
            print(f"OK - {item}")
        else:
            print(f"FAIL - {item} not found")
            all_ok = False
    
    return all_ok

def test_config():
    """测试配置"""
    print("\nTesting configuration...")
    
    try:
        from experiments.utils.config import get_default_config
        config = get_default_config()
        print(f"OK - Config loaded")
        print(f"  Environment: {config.experiment.env}")
        print(f"  Training episodes: {config.training.num_episodes}")
        return True
    except Exception as e:
        print(f"FAIL - Config error: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Multi-Agent Communication Framework - Test")
    print("=" * 60)
    
    tests = [
        ("Basic imports", test_basic),
        ("Project structure", test_project_structure),
        ("Configuration", test_config)
    ]
    
    results = []
    for name, test in tests:
        print(f"\n[Test] {name}")
        print("-" * 40)
        try:
            passed = test()
            results.append((name, passed))
        except Exception as e:
            print(f"Test error: {e}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed_count = 0
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:20} {status}")
        if passed:
            passed_count += 1
    
    total = len(results)
    print(f"\nPassed: {passed_count}/{total}")
    
    if passed_count == total:
        print("\nAll tests passed! Ready to run experiments.")
        return True
    else:
        print("\nSome tests failed. Please check installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)