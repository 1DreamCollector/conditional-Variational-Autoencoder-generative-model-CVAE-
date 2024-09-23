#!/usr/bin/env python
# coding=utf-8
'''
@Author ： LiJunXiong
@Institute : cumt
@Project :
@Data :
'''
import torch
import visdom

# 创建 Visdom 客户端
viz = visdom.Visdom()

# 创建一个矩阵
matrix = torch.randn(10, 10)

# 自定义热力图的 opts 参数
heatmap_opts = dict(
    colormap='plasma',  # 颜色映射为 Viridis
    xlabel='X Label',    # X 轴标签
    ylabel='Y Label',    # Y 轴标签
    title='Heatmap Title' # 标题
)

# 使用 visdom 的 viz.heatmap 函数画一个热力图，并自定义其 opts 参数
viz.heatmap(matrix, opts=heatmap_opts)