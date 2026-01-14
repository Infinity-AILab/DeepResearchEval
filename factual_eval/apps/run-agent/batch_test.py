#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量处理脚本:
1) 从指定文件夹中读取多个 JSON 文件
2) 从每个 JSON 文件中提取 user_query 和 response
3) 为每个文件构建最终 prompt 并执行 `uv run main.py fact-check`
4) 将运行结果重定向到对应的日志文件

Usage:
  python batch_test.py --json_dir /path/to/json/files --output_dir /path/to/logs
  python batch_test.py --json_dir ./json_files --output_dir ./logs
"""

import json
import argparse
import subprocess
from pathlib import Path
import sys
import os
from typing import Dict, Any, List, Tuple


def extract_query_and_response(json_file_path: Path) -> Tuple[str, str]:
    """
    从 JSON 文件中提取 user_query 和 response
    
    Args:
        json_file_path: JSON 文件路径
        
    Returns:
        Tuple[str, str]: (user_query, response)
        
    Raises:
        ValueError: 如果 JSON 文件格式不正确或缺少必要字段
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 尝试不同的 JSON 结构
        user_query = None
        response = None
        
        # 结构1: entries[0].query 和 entries[0].response
        if 'entries' in data and len(data['entries']) > 0:
            entry = data['entries'][0]
            user_query = entry.get('query', '')
            response = entry.get('response', '')
        
        # 结构2: 直接在根级别
        elif 'user_query' in data and 'response' in data:
            user_query = data['user_query']
            response = data['response']
        
        # 结构3: query 和 response 在根级别
        elif 'query' in data and 'response' in data:
            user_query = data['query']
            response = data['response']
        
        # 结构4: 检查是否有其他可能的字段名
        else:
            # 尝试找到包含 "query" 或 "question" 的字段
            for key, value in data.items():
                if isinstance(value, str) and ('query' in key.lower() or 'question' in key.lower()):
                    user_query = value
                    break
            
            # 尝试找到包含 "response" 或 "answer" 的字段
            for key, value in data.items():
                if isinstance(value, str) and ('response' in key.lower() or 'answer' in key.lower()):
                    response = value
                    break
        
        if not user_query or not response:
            raise ValueError(f"无法从 {json_file_path} 中提取 user_query 和 response")
        
        return user_query, response
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 文件格式错误 {json_file_path}: {e}")
    except Exception as e:
        raise ValueError(f"读取文件 {json_file_path} 时出错: {e}")


def process_single_json(json_file_path: Path, output_dir: Path, timeout: int = 3600, input_model: str = "") -> bool:
    """
    处理单个 JSON 文件
    
    Args:
        json_file_path: JSON 文件路径
        output_dir: 输出目录
        timeout: 超时时间（秒）
        
    Returns:
        bool: 处理是否成功
    """
    try:
        # 提取 user_query 和 response
        user_query, response = extract_query_and_response(json_file_path)
        
        # 生成任务 ID（基于文件名）
        task_id = "task_" + str(json_file_path.stem)
        
        # 构建日志文件路径
        log_file = output_dir / f"{task_id}.log"
        
        # 构建 uv 命令
        uv_cmd = [
            "uv", "run", "main.py", "fact-check",
            f"--task_id={task_id}",
            f"--task={response}",
            f"--user_query={user_query}",
            f"--input_model={input_model}",
        ]
        
        print(f"[INFO] 开始处理: {json_file_path.name} -> {log_file.name}")
        
        # 执行命令
        with open(log_file, "w", encoding="utf-8") as f:
            result = subprocess.run(
                uv_cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                cwd=Path(__file__).parent  # 确保在正确的目录下执行
            )
        
        if result.returncode == 0:
            print(f"[SUCCESS] 完成处理: {json_file_path.name}")
            return True
        else:
            print(f"[ERROR] 处理失败: {json_file_path.name} (返回码: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] 处理超时: {json_file_path.name} (超过 {timeout} 秒)")
        return False
    except Exception as e:
        print(f"[ERROR] 处理出错: {json_file_path.name} - {e}")
        return False

def process_directory(json_dir: Path, output_base_dir: Path, timeout: int, pattern: str) -> Tuple[int, int]:
    """
    处理单个目录
    
    Args:
        json_dir: JSON 文件目录路径
        output_base_dir: 输出基础目录
        timeout: 超时时间
        pattern: 文件匹配模式
        
    Returns:
        Tuple[int, int]: (成功数量, 失败数量)
    """
    input_model = json_dir.name   ## json_dir use /xx/xx  not /xx/xx/
    if type(input_model) != str:
        input_model = "input_model" + str(input_model)
    
    if not json_dir.exists():
        print(f"[ERROR] JSON 目录不存在: {json_dir}")
        return 0, 0
    
    if not json_dir.is_dir():
        print(f"[ERROR] 路径不是目录: {json_dir}")
        return 0, 0
    
    # 创建输出目录
    if input_model:
        output_dir = output_base_dir / input_model
    else:
        output_dir = output_base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找 JSON 文件
    json_files = list(json_dir.glob(pattern))
    
    if not json_files:
        print(f"[WARNING] 在 {json_dir} 中未找到匹配 {pattern} 的文件")
        return 0, 0
    
    print(f"[INFO] 找到 {len(json_files)} 个 JSON 文件")
    print(f"[INFO] 输出目录: {output_dir}")
    print(f"[INFO] 超时设置: {timeout} 秒")
    print("-" * 50)
    
    # 统计信息
    success_count = 0
    failed_count = 0
    
    # 处理每个 JSON 文件
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] 处理文件: {json_file.name}")
        
        if process_single_json(json_file, output_dir, timeout, input_model):
            success_count += 1
        else:
            failed_count += 1
        
        print("-" * 30)
    
    # 输出当前目录总结
    print(f"\n[SUMMARY] 目录 {json_dir.name} 处理完成:")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {failed_count} 个文件")
    print(f"  总计: {len(json_files)} 个文件")
    print(f"  日志目录: {output_dir}")
    print("=" * 60)
    
    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(description="批量处理 JSON 文件并执行 uv 命令")
    parser.add_argument("--json_dir", nargs='+', required=True, help="包含 JSON 文件的目录路径（支持多个目录）")
    parser.add_argument("--output_dir", default="../../logs", help="日志输出目录 (默认: ./logs)")
    parser.add_argument("--timeout", type=int, default=3600, help="单个任务超时时间（秒，默认: 3600）")
    parser.add_argument("--pattern", default="*.json", help="JSON 文件匹配模式 (默认: *.json)")
    # parser.add_argument("--max_files", type=int, help="最大处理文件数量（用于测试）")
    
    args = parser.parse_args()
    
    # 验证输入目录列表
    json_dirs = [Path(d) for d in args.json_dir]
    
    # 创建输出基础目录
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] 将处理 {len(json_dirs)} 个目录")
    print(f"[INFO] 基础输出目录: {output_base_dir}")
    print("=" * 60)
    
    # 总体统计信息
    total_success = 0
    total_failed = 0
    total_files = 0
    
    # 遍历处理每个目录
    for i, json_dir in enumerate(json_dirs, 1):
        print(f"\n[{i}/{len(json_dirs)}] 开始处理目录: {json_dir}")
        
        success_count, failed_count = process_directory(
            json_dir, output_base_dir, args.timeout, args.pattern
        )
        
        total_success += success_count
        total_failed += failed_count
        total_files += success_count + failed_count
    
    # 输出最终总结
    print(f"\n[FINAL SUMMARY] 所有目录处理完成:")
    print(f"  处理目录数: {len(json_dirs)}")
    print(f"  成功文件数: {total_success}")
    print(f"  失败文件数: {total_failed}")
    print(f"  总文件数: {total_files}")
    print(f"  基础日志目录: {output_base_dir}")

if __name__ == "__main__":
    main()
