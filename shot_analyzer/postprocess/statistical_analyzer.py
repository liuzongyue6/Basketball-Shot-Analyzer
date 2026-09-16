import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set_style("whitegrid")  # 设置seaborn样式
# 设置全局字体大小和中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 20})  # 设置全局字体大小


def plot_improvements(df, out_dir, show_plot=True):
    """A组/B组 CGSS 与 KCSS 改善值的箱形图和小提琴图，保存到 out_dir"""
    # 计算CGSS和KCSS的统一y轴范围
    cgss_min = df['CGSS_Improvement'].min()
    cgss_max = df['CGSS_Improvement'].max()
    kcss_min = df['KCSS_Improvement'].min()
    kcss_max = df['KCSS_Improvement'].max()

    # 统一y轴范围：取两者的最小值和最大值
    y_min = min(cgss_min, kcss_min)
    y_max = max(cgss_max, kcss_max)
    y_padding = (y_max - y_min) * 0.1  # 添加10%的边距
    y_lim = [y_min - y_padding, y_max + y_padding]

    # 设置图像大小
    plt.figure(figsize=(16, 8))  # 增大图像尺寸

    # CGSS_Improvement 的箱形图和散点图
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Group', y='CGSS_Improvement', data=df)
    sns.stripplot(x='Group', y='CGSS_Improvement', data=df, jitter=True, color='black', size=6, alpha=0.5)
    plt.title('A组和B组CGSS改善对比', fontsize=16)  # 增大标题字体
    plt.xlabel('组别', fontsize=14)  # 增大x轴标签字体
    plt.ylabel('CGSS改善值', fontsize=14)  # 增大y轴标签字体
    plt.tick_params(axis='both', which='major', labelsize=12)  # 增大刻度标签字体
    plt.ylim(y_lim)  # 设置统一的y轴范围

    # KCSS_Improvement 的箱形图和散点图
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Group', y='KCSS_Improvement', data=df)
    sns.stripplot(x='Group', y='KCSS_Improvement', data=df, jitter=True, color='black', size=6, alpha=0.5)
    plt.title('A组和B组KCSS改善对比', fontsize=16)  # 增大标题字体
    plt.xlabel('组别', fontsize=14)  # 增大x轴标签字体
    plt.ylabel('KCSS改善值', fontsize=14)  # 增大y轴标签字体
    plt.tick_params(axis='both', which='major', labelsize=12)  # 增大刻度标签字体
    plt.ylim(y_lim)  # 设置统一的y轴范围

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'improvement_comparison_boxplot.png'), dpi=300)  # 增加分辨率
    if show_plot: plt.show()


    # 使用小提琴图来更好地观察数据分布
    plt.figure(figsize=(16, 8))  # 增大图像尺寸

    # CGSS_Improvement 的小提琴图
    plt.subplot(1, 2, 1)
    sns.violinplot(x='Group', y='CGSS_Improvement', data=df, inner='quartile')
    sns.stripplot(x='Group', y='CGSS_Improvement', data=df, jitter=True, color='black', size=6, alpha=0.5)
    plt.title('A组与B组CGSS改善分布', fontsize=16)  # 增大标题字体
    plt.xlabel('组别', fontsize=14)  # 增大x轴标签字体
    plt.ylabel('CGSS改善值', fontsize=14)  # 增大y轴标签字体
    plt.tick_params(axis='both', which='major', labelsize=12)  # 增大刻度标签字体
    plt.ylim(y_lim)  # 设置统一的y轴范围

    # KCSS_Improvement 的小提琴图
    plt.subplot(1, 2, 2)
    sns.violinplot(x='Group', y='KCSS_Improvement', data=df, inner='quartile')
    sns.stripplot(x='Group', y='KCSS_Improvement', data=df, jitter=True, color='black', size=6, alpha=0.5)
    plt.title('A组与B组KCSS改善分布', fontsize=16)  # 增大标题字体
    plt.xlabel('组别', fontsize=14)  # 增大x轴标签字体
    plt.ylabel('KCSS改善值', fontsize=14)  # 增大y轴标签字体
    plt.tick_params(axis='both', which='major', labelsize=12)  # 增大刻度标签字体
    plt.ylim(y_lim)  # 设置统一的y轴范围

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'improvement_comparison_violinplot.png'), dpi=300)  # 增加分辨率
    if show_plot: plt.show()


def compare_groups(df, alpha=0.05):
    """A组与B组改善值的独立样本t检验"""
    # 将A组和B组的数据分开
    group_a = df[df['Group'] == 'A']
    group_b = df[df['Group'] == 'B']

    # 对 CGSS_Improvement 进行独立样本t检验
    t_stat_cgss, p_value_cgss = stats.ttest_ind(group_a['CGSS_Improvement'], group_b['CGSS_Improvement'])

    # 对 KCSS_Improvement 进行独立样本t检验
    t_stat_kcss, p_value_kcss = stats.ttest_ind(group_a['KCSS_Improvement'], group_b['KCSS_Improvement'])

    # 打印检验结果
    print(f"T-test for CGSS_Improvement: t-statistic = {t_stat_cgss}, p-value = {p_value_cgss}")
    print(f"T-test for KCSS_Improvement: t-statistic = {t_stat_kcss}, p-value = {p_value_kcss}")


    # --- 结论分析 ---
    print("\n--- 统计显著性分析 ---")
    if p_value_cgss < alpha:
        print("A组和B组在CGSS_Improvement上的差异是统计显著的。")
    else:
        print("A组和B组在CGSS_Improvement上的差异不具备统计显著性。")

    if p_value_kcss < alpha:
        print("A组和B组在KCSS_Improvement上的差异是统计显著的。")
    else:
        print("A组和B组在KCSS_Improvement上的差异不具备统计显著性。")


def main():
    parser = argparse.ArgumentParser(description="Compare CGSS/KCSS improvement between group A and group B.")
    parser.add_argument("csv", help="group data CSV with Group, CGSS_Improvement and KCSS_Improvement columns")
    parser.add_argument("--out-dir", help="where to save the box and violin plots (default: the CSV's folder)")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    args = parser.parse_args()

    # 加载数据集
    df = pd.read_csv(args.csv)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    plot_improvements(df, out_dir, show_plot=not args.no_show)
    compare_groups(df)


if __name__ == "__main__":
    main()
