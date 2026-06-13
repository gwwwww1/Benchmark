import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast
from contextlib import suppress
from tqdm import tqdm
import torch.nn.functional as F


def evaluate(model, dataloader, tokenizer, device, amp=True, recall_k_list=[5]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    Returns
    -------

    dict of accuracy metric
    """
    # 根据 amp 参数选择是否使用自动混合精度
    # 如果 amp 为 True，则使用 torch.cuda.amp.autocast；否则使用 suppress 上下文管理器（不做任何操作）
    autocast = torch.cuda.amp.autocast if amp else suppress
    # 用于存储每个样本的预测结果
    preds = []
    # 使用 tqdm 显示数据加载的进度条
    for batch_images, batch_texts in tqdm(dataloader):
        # 将批量图像数据移动到指定设备（CPU 或 GPU）上
        batch_images = batch_images.to(device)
        # 对批量中的所有文本进行分词处理
        # 首先将批量中的所有文本展平为一个列表，然后使用 tokenizer 进行分词，最后将分词结果移动到指定设备上
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # 记录每个图像对应的文本数量
        nb_texts_for_each_image = [len(texts) for texts in batch_texts]

        # 在无梯度计算的上下文中进行推理，以节省内存和计算资源
        # 如果使用自动混合精度，则在 autocast 上下文中进行计算
        with torch.no_grad(), autocast():
            # 对批量图像进行编码，并进行 L2 归一化处理，最后将结果移动到 CPU 上
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1).cpu()
            # 对批量文本进行编码，并进行 L2 归一化处理，最后将结果移动到 CPU 上
            batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1).cpu()
        # 初始化文本嵌入的起始索引
        start = 0
        # 遍历每个图像及其对应的文本数量
        for i, nb in enumerate(nb_texts_for_each_image):
            # 计算当前图像对应的文本嵌入的结束索引
            end = start + nb
            # 获取当前图像的嵌入向量
            image_emb = batch_images_emb[i:i + 1]
            # 获取当前图像对应的所有文本的嵌入向量
            texts_emb = batch_texts_emb[start:end]
            # 计算图像嵌入向量与文本嵌入向量的点积，得到相似度分数
            scores = image_emb @ texts_emb.t()
            # 取出第一个元素（因为只有一个图像嵌入）
            scores = scores[0]
            # 找到相似度分数最大的文本的索引
            pred = scores.argmax().item()
            # 更新起始索引，为下一个图像的文本处理做准备
            start = end
            # 将预测结果添加到 preds 列表中
            preds.append(pred)
    # 将预测结果列表转换为长整型的 PyTorch 张量
    pred = torch.Tensor(preds).long()
    # 计算预测结果中索引为 0 的样本的比例，即准确率
    # 这里假设索引为 0 的文本是正确的文本，其余文本为负样本
    acc = (pred == 0).float().mean().item()
    # 初始化一个空字典，用于存储评估指标
    metrics = {}
    # 将准确率存储到 metrics 字典中
    metrics[f"acc"] = acc
    # 返回包含评估指标的字典
    return metrics