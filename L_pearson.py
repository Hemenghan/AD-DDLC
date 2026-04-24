import torch
import torch.nn.functional as F

def pearson_correlation(x, y, dim=1, eps=1e-8):
    """
    计算两个张量之间的Pearson相关系数（向量化实现）
    
    Args:
        x (torch.Tensor): 输入张量1 [batch_size, num_classes]
        y (torch.Tensor): 输入张量2 [batch_size, num_classes]
        dim (int): 计算相关性的维度
        eps (float): 防止除零的小数值
    
    Returns:
        torch.Tensor: Pearson相关系数 [batch_size,]
    """
    # 中心化数据
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    
    # 计算协方差
    covariance = (x_centered * y_centered).sum(dim=dim)
    
    # 计算标准差
    x_std = torch.sqrt((x_centered ** 2).sum(dim=dim) + eps)
    y_std = torch.sqrt((y_centered ** 2).sum(dim=dim) + eps)
    
    # 计算相关系数
    correlation = covariance / (x_std * y_std)
    return correlation

def pearson_loss(x, y, dim=1, eps=1e-8):
    """
    基于Pearson相关系数的损失函数
    损失 = 1 - Pearson相关系数，完美正相关时损失为0
    
    Args:
        x (torch.Tensor): 学生模型输出 [batch_size, num_classes]
        y (torch.Tensor): 教师模型输出 [batch_size, num_classes]
        dim (int): 计算相关性的维度
        eps (float): 防止除零的小数值
    
    Returns:
        torch.Tensor: Pearson损失值
    """
    # 计算批次中每个样本的Pearson相关系数
    r = pearson_correlation(x, y, dim=dim, eps=eps)
    
    # 损失为1 - 相关系数，然后求批次平均
    loss = (1 - r).mean()
    return loss