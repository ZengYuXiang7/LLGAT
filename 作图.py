# coding : utf-8
# Author : Anonymous
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
from matplotlib import rcParams
# 设置中文字体为你本机支持的字体之一
rcParams['font.family'] = 'Heiti TC'  # 可替换为 'PingFang HK' 等
rcParams['axes.unicode_minus'] = False

def ablation_draw_4_bar_plots_with_error(data, datasets, methods, x_labels, y_labels, idx_names, colors, file_name):
    fig, axs = plt.subplots(1, len(datasets) * 2, figsize=(5 * len(datasets) + 4, 3.5), dpi=600)
    fontsize = 16
    font_dict = {'fontname': 'Heiti TC', 'fontsize': fontsize, 'fontweight': 'bold'}  # Font settings
    bar_width = 0.30
    default_error = 0.001  # Default error for all bars
    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            for k, metric in enumerate(data[i][j]):
                print(method, dataset, metric)
                axs[j * len(datasets) + k].bar(
                    np.arange(len(metric)) + i * bar_width,
                    metric,
                    # yerr=default_error,  # Adding error bars
                    capsize=2,  # Size of the error bar caps
                    width=bar_width,
                    label=methods[i],
                    color=colors[i],
                    edgecolor='black',  # Set edge color to black
                    linewidth=1,  # Set edge line width
                )
                # Set x and y axis labels with bold font
                axs[j * len(datasets) + k].set_xlabel(x_labels, fontdict=font_dict)
                axs[j * len(datasets) + k].set_ylabel(y_labels[k], fontdict=font_dict)
                axs[j * len(datasets) + k].set_title(datasets[j], fontdict=font_dict)

                # Set x-axis tick labels
                axs[j * len(datasets) + k].set_xticks(np.arange(len(metric)) + (len(methods) - 1) / 2 * bar_width)
                axs[j * len(datasets) + k].set_xticklabels(
                    idx_names,
                    rotation=0,
                    fontname='Times New Roman',
                    fontsize=fontsize,
                    fontweight='bold'
                )

                # Set y-axis tick labels
                for tick in axs[j * len(datasets) + k].get_yticklabels():
                    tick.set_fontname('Times New Roman')
                    tick.set_fontsize(fontsize)
                    tick.set_fontweight('bold')

                # Set grid behind bars
                axs[j * len(datasets) + k].set_axisbelow(True)
                axs[j * len(datasets) + k].grid(True, linestyle='-', alpha=0.2)

    plt.tight_layout(rect=[0, -0.02, 0.98, 0.90])  # Adjust layout
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[:len(methods)], labels[:len(methods)], loc='upper center', ncol=3, bbox_to_anchor=(0.524, 0.97))

    # Save the plot as a PDF
    os.makedirs('figs', exist_ok=True)
    pdf_path = f'./figs/{file_name}.pdf'
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.show()
    print(f"The plots with error bars have been saved to {pdf_path}")
    return pdf_path


def ablation_draw_4_line_plots(data, datasets, methods, x_labels, y_labels, idx_names, colors, file_name):
    # fig, axs = plt.subplots(1, 4, figsize=(14, 3.5), dpi=600)
    fig, axs = plt.subplots(1, len(datasets) * 2, figsize=(5 * len(datasets) + 4, 3.5), dpi=600)
    markers = ['o', 'x', '*', 's', 'd', '^']  # 圆圈, 叉叉, 星星, 方块, 菱形, 上三角等标记
    fontsize = 16
    font_dict = {'fontname': 'Heiti TC', 'fontsize': fontsize, 'fontweight': 'bold'}  # 定义字体字典
    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            for k, metric in enumerate(data[i][j]):
                print(method, dataset, metric)
                axs[j * len(datasets) + k].plot(
                    metric,
                    label=methods[i],
                    color=colors[i],
                    marker=markers[i],
                    markerfacecolor='none',
                )
                # 设置 x 和 y 轴标签，加粗字体
                axs[j * len(datasets) + k].set_xlabel(x_labels, fontdict=font_dict)
                axs[j * len(datasets) + k].set_ylabel(y_labels[k], fontdict=font_dict)
                # 设置 x 轴刻度标签字体
                axs[j * len(datasets) + k].set_xticks(np.arange(len(idx_names)))
                axs[j * len(datasets) + k].set_xticklabels(
                    idx_names,
                    rotation=0,
                    fontname='Times New Roman',
                    fontsize=fontsize,
                    fontweight='bold'
                )
                # fixed_error = 0.005
                # Add the shaded error region
                # axs[j * len(datasets) + k].fill_between(
                #     range(len(metric)),  # X-axis
                #     metric - fixed_error,  # Lower bound
                #     metric + fixed_error,  # Upper bound
                #     color=colors[k],
                #     alpha=0.2  # Transparency of the shaded area
                # )
                axs[j * len(datasets) + k].set_title(datasets[j], fontdict=font_dict)

                # 设置 y 轴刻度标签字体
                for tick in axs[j * len(datasets) + k].get_yticklabels():
                    tick.set_fontname('Times New Roman')
                    tick.set_fontsize(fontsize)
                    tick.set_fontweight('bold')

                # Set grid behind bars
                axs[j * len(datasets) + k].set_axisbelow(True)
                axs[j * len(datasets) + k].grid(True, linestyle='-', alpha=0.2)

    plt.tight_layout(rect=[0, -0.02, 0.98, 0.90])  # [left, bottom, right, top]
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[:len(methods)], labels[:len(methods)], loc='upper center', ncol=3, bbox_to_anchor=(0.524, 0.97))

    os.makedirs('figs', exist_ok=True)
    pdf_path = f'./figs/{file_name}.pdf'
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.show()

    print(f"The plots have been saved to {pdf_path}")
    return pdf_path


def ablation1(df):
    #
    df[0] = [
        # 第一个数据集
        [
            [0.0831, 0.0656, 0.0559, 0.0519, 0.0452],  # NMAE
            [0.1096, 0.0947, 0.0877, 0.0863, 0.0775]  # NRMSE
        ],
        # 第二个数据集
        [
            [0.0830, 0.0654, 0.0545, 0.0508, 0.0458],  # NMAE
            [0.0958, 0.0863, 0.0863, 0.0792, 0.0655]  # NRMSE
        ]
    ]
    #
    df[1] = [
        # 第一个数据集
        [
            [0.0685, 0.0552, 0.0532, 0.0450, 0.0363],  # NMAE
            [0.0958, 0.0863, 0.0863, 0.0792, 0.0655]  # NRMSE
        ],
        # 第二个数据集
        [
            [0.0630, 0.0534, 0.0441, 0.0419, 0.0354],  # NMAE
            [0.0895, 0.0834, 0.0751, 0.0736, 0.0636]  # NRMSE
        ]
    ]
    df[2] = [
        # 第一个数据集
        [
            [0.0459, 0.0316, 0.0243, 0.0236, 0.0207],  # NMAE
            [0.0662, 0.0464, 0.0364, 0.0360, 0.0321]  # NRMSE
        ],
        # 第二个数据集
        [
            [0.0468, 0.0346, 0.0241, 0.0235, 0.0198],  # NMAE
            [0.0695, 0.0531, 0.0352, 0.0344, 0.0294]  # NRMSE
        ]
    ]
    return df


def ablation2(df):
    #
    df[0] = [
        # 第一个数据集
        [
            [0.0869, 0.0686, 0.0527, 0.0388, 0.0332],  # NMAE
            [0.1217, 0.0943, 0.0840, 0.0691, 0.0619]  # NRMSE
        ],
        # 第二个数据集
        [
            [0.0661, 0.0531, 0.0421, 0.0396, 0.0345],  # NMAE
            [0.1064, 0.0888, 0.0726, 0.0710, 0.0643]  # NRMSE
        ]
    ]
    #
    df[1] = [
        # 第一个数据集
        [
            [0.0810, 0.0640, 0.0486, 0.0335, 0.0279],  # NMAE
            [0.1135, 0.0900, 0.0767, 0.0607, 0.0510]  # NRMSE
        ],
        # 第二个数据集
        [
            [0.0635, 0.0504, 0.0399, 0.0372, 0.0318],  # NMAE
            [0.0922, 0.0766, 0.0647, 0.0619, 0.0565]  # NRMSE
        ]
    ]
    df[2] = [
        # 第一个数据集
        [
            [0.0459, 0.0316, 0.0243, 0.0236, 0.0207],  # NMAE
            [0.0662, 0.0464, 0.0364, 0.0360, 0.0321]  # NRMSE
        ],
        # 第二个数据集
        [
            [0.0468, 0.0346, 0.0241, 0.0235, 0.0198],  # NMAE
            [0.0695, 0.0531, 0.0352, 0.0344, 0.0294]  # NRMSE
        ]
    ]
    return df


def ablation3(df):
    #
    df[0] = [
        # 第一个数据集
        [
            [0.0578, 0.0485, 0.0350, 0.0334, 0.0223],  # NMAE
            [0.0846, 0.0700, 0.0586, 0.0577, 0.0356]  # NRMSE
        ],
        # 第二个数据集
        [
            [0.0569, 0.0461, 0.0349, 0.0305, 0.0207],  # NMAE
            [0.0844, 0.0732, 0.0568, 0.0499, 0.0320]  # NRMSE
        ]
    ]
    #
    df[1] = [
        # 第一个数据集
        [
            [0.0459, 0.0316, 0.0243, 0.0236, 0.0207],  # NMAE
            [0.0662, 0.0464, 0.0364, 0.0360, 0.0321]  # NRMSE
        ],
        # 第二个数据集
        [
            [0.0468, 0.0346, 0.0241, 0.0235, 0.0198],  # NMAE
            [0.0695, 0.0531, 0.0352, 0.0344, 0.0294]  # NRMSE
        ]
    ]
    return df

def ablation1_config():
    # 消融实验1：不同输入特征编码方式对比
    methods = ['基于数值编码', '基于独热编码', '基于独热编码']
    idx_names = ['100', '200', '400', '500', '900']  # 样本数量
    datasets = ['CPU数据集', 'GPU数据集']
    x_labels = '样本数量'
    y_labels = ['归一化平均绝对误差', '归一化均方根误差']
    return datasets, methods, x_labels, y_labels, idx_names

def ablation2_config():
    # 消融实验2：不同图神经网络架构对比
    methods = ['基于GCN', '基于GraphSage', '基于GAT']
    idx_names = ['100', '200', '400', '500', '900']  # 样本数量
    datasets = ['CPU数据集', 'GPU数据集']
    x_labels = '样本数量'
    y_labels = ['归一化平均绝对误差', '归一化均方根误差']
    return datasets, methods, x_labels, y_labels, idx_names

def ablation3_config():
    # 消融实验3：是否引入大语言模型（LLM）对比
    methods = ['使用LLM', '不使用LLM']
    idx_names = ['100', '200', '400', '500', '900']  # 样本数量
    datasets = ['CPU数据集', 'GPU数据集']
    x_labels = '样本数量'
    y_labels = ['归一化平均绝对误差', '归一化均方根误差']
    return datasets, methods, x_labels, y_labels, idx_names


if __name__ == '__main__':
    # 消融实验1
    datasets, methods, x_labels, y_labels, idx_names = ablation1_config()
    df = np.zeros((len(methods), len(datasets), len(y_labels), len(idx_names)))
    print(df.shape)
    df = ablation1(df)
    colors = ['#43747B', '#679DBD', '#f2d2bb']
    ablation_draw_4_bar_plots_with_error(df, datasets, methods, x_labels, y_labels, idx_names, colors, 'ablation1')

    # 消融实验2
    datasets, methods, x_labels, y_labels, idx_names = ablation2_config()
    df = np.zeros((len(methods), len(datasets), len(y_labels), len(idx_names)))
    print(df.shape)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    df = ablation2(df)
    ablation_draw_4_line_plots(df, datasets, methods, x_labels, y_labels, idx_names, colors, 'ablation2')

    # 消融实验3
    datasets, methods, x_labels, y_labels, idx_names = ablation3_config()
    df = np.zeros((len(methods), len(datasets), len(y_labels), len(idx_names)))
    print(df.shape)
    df = ablation3(df)
    colors = ['#069af3', '#ffa500']
    ablation_draw_4_bar_plots_with_error(df, datasets, methods, x_labels, y_labels, idx_names, colors, 'ablation3')
