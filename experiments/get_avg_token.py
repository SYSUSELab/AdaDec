import os
import re
import pandas as pd
from pathlib import Path

def count_steps_in_log(log_file_path):
    """
    统计log文件中STEP的数量
    
    Args:
        log_file_path: log文件的完整路径
        
    Returns:
        int: STEP的数量，如果文件读取失败则返回None
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 使用正则表达式查找所有STEP
        step_pattern = re.compile(r'STEP:(\d+)')
        matches = step_pattern.findall(content)
        
        # 返回匹配的数量
        return len(matches)
        
    except Exception as e:
        print(f"读取文件 {log_file_path} 时出错: {e}")
        return None

def scan_log_files_for_steps(root_folder):
    """
    扫描指定文件夹下的所有log文件并统计STEP数量
    
    Args:
        root_folder: 根文件夹路径
        
    Returns:
        list: 包含统计结果的字典列表
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
                        step_count = count_steps_in_log(log_file)
                        
                        results.append({
                            '模型': model_name,
                            '方法': method_name,
                            'STEP数量(Token数)': step_count if step_count is not None else 'N/A',
                            '日志文件路径': str(log_file)
                        })
                        
                        # 打印统计进度
                        status = str(step_count) if step_count is not None else "读取失败"
                        print(f"已处理: {model_name}/{method_name} - STEP数量: {status}")
                    else:
                        print(f"警告: 未找到文件 {log_file}")
    
    return results

def analyze_step_statistics(df):
    """
    分析STEP统计数据
    
    Args:
        df: 包含统计结果的DataFrame
    """
    # 过滤出有效数据
    valid_data = df[df['STEP数量(Token数)'] != 'N/A'].copy()
    valid_data['STEP数量(Token数)'] = valid_data['STEP数量(Token数)'].astype(int)
    
    if len(valid_data) == 0:
        print("没有有效的STEP数据进行统计分析")
        return
    
    print(f"\n统计分析:")
    print(f"总文件数: {len(df)}")
    print(f"成功统计: {len(valid_data)}")
    print(f"平均Token数: {valid_data['STEP数量(Token数)'].mean():.0f}")
    print(f"最少Token数: {valid_data['STEP数量(Token数)'].min()}")
    print(f"最多Token数: {valid_data['STEP数量(Token数)'].max()}")
    print(f"Token数总和: {valid_data['STEP数量(Token数)'].sum()}")
    
    # 按模型分组统计
    if len(valid_data) > 1:
        print(f"\n按模型分组统计:")
        model_stats = valid_data.groupby('模型')['STEP数量(Token数)'].agg(['count', 'mean', 'sum']).round(0)
        model_stats.columns = ['文件数', '平均Token数', 'Token总数']
        print(model_stats.to_string())
        
        # 按方法分组统计
        print(f"\n按方法分组统计:")
        method_stats = valid_data.groupby('方法')['STEP数量(Token数)'].agg(['count', 'mean', 'sum']).round(0)
        method_stats.columns = ['文件数', '平均Token数', 'Token总数']
        print(method_stats.to_string())

def main():
    """
    主函数
    """
    # 指定根文件夹路径（可以修改为实际路径）
    root_folder = "mbpp+_outputs"
    
    print(f"开始扫描文件夹: {root_folder}")
    print("正在统计每个log文件中的STEP数量...")
    print("-" * 60)
    
    # 统计STEP数量
    results = scan_log_files_for_steps(root_folder)
    
    if not results:
        print("未找到任何数据")
        return
    
    # 创建DataFrame并显示表格
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("STEP统计结果汇总表格:")
    print("=" * 80)
    
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 60)
    
    # 按模型和方法排序
    df_sorted = df.sort_values(['模型', '方法'])
    print(df_sorted.to_string(index=False))
    
    # 保存到CSV文件
    output_file = "step_count_results.csv"
    df_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")
    
    # 进行统计分析
    analyze_step_statistics(df)

# 如果需要只统计特定模型或方法，可以使用以下函数
def filter_results(root_folder, target_models=None, target_methods=None):
    """
    过滤特定模型或方法的结果
    
    Args:
        root_folder: 根文件夹路径
        target_models: 目标模型列表，如 ['codellama-7b', 'gpt-3.5']
        target_methods: 目标方法列表，如 ['AdaDynL', 'Traditional']
        
    Returns:
        DataFrame: 过滤后的结果
    """
    results = scan_log_files_for_steps(root_folder)
    df = pd.DataFrame(results)
    
    if target_models:
        df = df[df['模型'].isin(target_models)]
    
    if target_methods:
        df = df[df['方法'].isin(target_methods)]
    
    return df

if __name__ == "__main__":
    main()
    
    # 示例：如果只想统计特定模型和方法的数据
    # print("\n" + "=" * 50)
    # print("过滤示例 - 只统计codellama-7b模型的AdaDynL和Traditional方法:")
    # filtered_df = filter_results("humaneval+_outputs", 
    #                             target_models=['codellama-7b'], 
    #                             target_methods=['AdaDynL', 'Traditional'])
    # print(filtered_df.to_string(index=False))