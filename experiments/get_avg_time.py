import os
import re
import pandas as pd
from pathlib import Path

def extract_time_from_log(log_file_path):
    """
    从log文件的倒数10行中提取时间数据
    
    Args:
        log_file_path: log文件的完整路径
        
    Returns:
        float: 提取到的时间数据，如果未找到则返回None
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        # 获取倒数10行
        last_10_lines = lines[-10:] if len(lines) >= 10 else lines
        
        # 在倒数10行中查找时间数据
        pattern = r'\[INFO\].*?Total time taken:\s*(\d+\.?\d*)\s*seconds'
        
        for line in reversed(last_10_lines):  # 从最后一行开始查找
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))
                
    except Exception as e:
        print(f"读取文件 {log_file_path} 时出错: {e}")
    
    return None

def scan_log_files(root_folder):
    """
    扫描指定文件夹下的所有log文件并提取时间数据
    
    Args:
        root_folder: 根文件夹路径
        
    Returns:
        list: 包含提取结果的字典列表
    """
    results = []
    root_path = Path(root_folder)
    
    if not root_path.exists():
        print(f"错误: 文件夹 {root_folder} 不存在")
        return results
    
    # 遍历所有模型文件夹
    for model_folder in root_path.iterdir():
        if model_folder.is_dir():
            model_name = model_folder.name
            
            # 遍历每个模型下的方法文件夹
            for method_folder in model_folder.iterdir():
                if method_folder.is_dir():
                    method_name = method_folder.name
                    
                    # 查找log文件
                    log_file = method_folder / f"{model_name}.log"
                    
                    if log_file.exists():
                        time_data = extract_time_from_log(log_file)
                        
                        # 计算每一题所花的平均时间
                        # if root_folder == 'humaneval+_outputs':
                        #     time_data = time_data / 164 if time_data is not None else None
                        # elif root_folder == 'mbpp+_outputs':
                        #     time_data = time_data / 378 if time_data is not None else None
                        # else:
                        #     print(f"警告: 未知的 {root_folder}，未进行时间平均化处理")
                        
                        results.append({
                            '模型': model_name,
                            '方法': method_name,
                            '时间(秒)': time_data if time_data is not None else 'N/A',
                            '日志文件路径': str(log_file)
                        })
                        
                        # 打印提取进度
                        status = f"{time_data:.2f}" if time_data is not None else "未找到"
                        print(f"已处理: {model_name}/{method_name} - 时间: {status}")
                    else:
                        print(f"警告: 未找到文件 {log_file}")
    
    return results

def main():
    """
    主函数
    """
    # 指定根文件夹路径
    root_folder = "mbpp+_outputs"  # 可以修改为实际路径
    
    print(f"开始扫描文件夹: {root_folder}")
    print("-" * 50)
    
    # 提取数据
    results = scan_log_files(root_folder)
    
    if not results:
        print("未找到任何数据")
        return
    
    # 创建DataFrame并显示表格
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 50)
    print("提取结果汇总表格:")
    print("=" * 50)
    
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(df.to_string(index=False))
    
    # 保存到CSV文件
    output_file = "time_extraction_results.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")
    
    # 统计信息
    valid_times = df[df['时间(秒)'] != 'N/A']['时间(秒)'].astype(float)
    if len(valid_times) > 0:
        print(f"\n统计信息:")
        print(f"总文件数: {len(results)}")
        print(f"成功提取: {len(valid_times)}")
        # print(f"平均时间: {valid_times.mean():.2f} 秒")
        # print(f"最短时间: {valid_times.min():.2f} 秒")
        # print(f"最长时间: {valid_times.max():.2f} 秒")

if __name__ == "__main__":
    main()