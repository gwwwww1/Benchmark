"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import csv
import logging
import pdb
from contextlib import suppress
import os
import torch
import torch.nn.functional as F
from torch import autocast
from tqdm import tqdm
import numpy as np
import glob
from sklearn.metrics import classification_report, balanced_accuracy_score

from clip_benchmark.metrics.zeroshot_retrieval import get_text_embeddings, get_image_embeddings

import numpy as np
from sklearn.metrics import accuracy_score


# bootstrap_metric抽样置信95%
def bootstrap_metric(y_true, y_pred, metric_func=accuracy_score, n_bootstrap=1000, confidence_level=0.95):
    """
    使用 Bootstrap 方法估计指标的置信区间。

    参数:
    y_true (array-like): 真实标签
    y_pred (array-like): 预测标签
    metric_func (callable): 要估计的指标函数，默认为准确率
    n_bootstrap (int): 抽样次数，默认为 1000
    confidence_level (float): 置信水平，默认为 0.95

    返回:
    tuple: 置信区间的下限和上限
    """
    metrics = []
    data_size = len(y_true)
    for _ in range(n_bootstrap):
        # 有放回抽样
        indices = np.random.choice(data_size, size=data_size, replace=True)
        sample_y_true = np.array(y_true)[indices]
        sample_y_pred = np.array(y_pred)[indices]
        # 计算指标
        metric = metric_func(sample_y_true, sample_y_pred)
        metrics.append(metric)

    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    # 计算置信区间
    lower_bound = np.percentile(metrics, lower_percentile)
    upper_bound = np.percentile(metrics, upper_percentile)

    return lower_bound, upper_bound


def bootstrap_metric_2(y_true, y_pred, metric_func=accuracy_score, metric_func_2=accuracy_score, n_bootstrap=1000,
                       confidence_level=0.95):
    """
    使用 Bootstrap 方法估计指标的置信区间。

    参数:
    y_true (array-like): 真实标签
    y_pred (array-like): 预测标签
    metric_func (callable): 要估计的指标函数，默认为准确率
    n_bootstrap (int): 抽样次数，默认为 1000
    confidence_level (float): 置信水平，默认为 0.95

    返回:
    tuple: 置信区间的下限和上限
    """
    metrics, metrics_2 = [], []
    data_size = len(y_true)
    for _ in range(n_bootstrap):
        # 有放回抽样
        indices = np.random.choice(data_size, size=data_size, replace=True)
        sample_y_true = np.array(y_true)[indices]
        sample_y_pred = np.array(y_pred)[indices]
        # 计算指标
        metric = metric_func(sample_y_true, sample_y_pred)
        metric_2 = metric_func_2(sample_y_true, sample_y_pred)
        metrics.append(metric)
        metrics_2.apend(metric_2)

    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    # 计算置信区间
    lower_bound = np.percentile(metrics, lower_percentile)
    upper_bound = np.percentile(metrics, upper_percentile)
    lower_bound_2 = np.percentile(metrics_2, lower_percentile)
    upper_bound_2 = np.percentile(metrics_2, upper_percentile)

    return lower_bound, upper_bound, lower_bound_2, upper_bound_2


def xlm_tokenizer(tokens, tokenizer, max_len=100):
    tokens = tokenizer.encode(tokens)

    tokens = tokens[1:-1]  # remove eos and bos;
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]

    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]  # add eos and bos
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)

    text_tokens = tokens + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return text_tokens, padding_mask


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []

        for classname in tqdm(classnames):

            if type(templates) == dict:
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif type(templates) == list:
                # generic prompts tht are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")

            # MUSK tokenizer for encoding class names
            if tokenizer.__class__.__name__ == "XLMRobertaTokenizer":

                text_ids = []
                paddings = []
                for txt in texts:
                    txt_ids, pad = xlm_tokenizer(txt, tokenizer, max_len=100)
                    text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
                    paddings.append(torch.tensor(pad).unsqueeze(0))

                text_ids = torch.cat(text_ids)
                paddings = torch.cat(paddings)
                class_embeddings = model(
                    text_description=text_ids.to(device),
                    padding_mask=paddings.to(device),
                    with_head=True,
                    out_norm=True
                )[1]

                class_embedding = class_embeddings.mean(dim=0)
                # class_embedding /= class_embedding.norm()

            elif tokenizer.__class__.__name__ == "CLIPTokenizerFast":
                inputs = tokenizer(texts, padding=True, return_tensors="pt")
                text_features = model.get_text_features(inputs['input_ids'].to(device),
                                                        inputs['attention_mask'].to(device))

                class_embedding = text_features.mean(dim=0)
                class_embedding /= class_embedding.norm()

            # tokenizer for CONCH
            elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
                from clip_benchmark.models.conch.open_clip_custom import tokenize
                tokenized_prompts = tokenize(texts=texts, tokenizer=tokenizer).to(device)
                text_features = model.encode_text(tokenized_prompts)
                class_embedding = text_features.mean(dim=0)
                class_embedding /= class_embedding.norm()

            else:

                texts = tokenizer(texts).to(device)  # tokenize
                class_embeddings = model.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, amp=True):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0

    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                model_name = model.__class__.__name__.lower()
                # print(model_name)

                # predict for musk model
                if 'musk' in model_name:
                    image_features = model(
                        image=images,
                        out_norm=True,
                        with_head=True  # head must be used for zero-shot task
                    )[0]

                elif 'clipmodel' in model_name:
                    image_features = model.get_image_features(images)

                # image embeddings for CONCH
                elif 'CoCa' in model_name:
                    """
                    https://github.com/mahmoodlab/CONCH
                    """
                    image_features = model.encode_image(
                        images,
                        proj_contrast=True,
                        normalize=True
                    )

                # predict for clip model
                else:
                    image_features = model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
                # logits = 100. * image_features @ classifier
                logits = 100. * image_features @ classifier
            true.append(target.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true


def Acc(y_true, y_pred):
    """
    此函数用于计算模型预测的准确率
    :param y_true: 真实标签，为一维数组
    :param y_pred: 预测标签，为一维数组
    :return: 准确率
    """
    # 检查真实标签和预测标签的长度是否一致
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签的长度必须一致。")
    # 计算预测正确的样本数量
    correct_predictions = np.sum(y_true == y_pred)
    # 计算准确率
    return correct_predictions / len(y_true)


def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, save_clf=None,
             compare=True,
             load_clfs=[]):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        # allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为 MB
        # print(f"当前已分配的显存: {allocated_memory:.2f} MB")
        # print(model)
        model = model.to(device)
        # allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为 MB
        # print(f"当前已分配的显存: {allocated_memory:.2f} MB")
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)

    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target = run_classification(model, classifier, dataloader, device, amp=amp)
    # allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为 MB
    # print(f"当前已分配的显存: {allocated_memory:.2f} MB")
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # print(pred)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan")
        # 召回率的平均值
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        acc = Acc(target, pred)
        # with open('output.csv', 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     # 写入表头
        #     writer.writerow(['Ground Truth', 'Prediction'])
        #     # 逐行写入数据
        #     for gt, pred in zip(target, pred):
        #         writer.writerow([gt, pred])
        print(acc)
        if verbose:
            print(classification_report(target, pred, digits=3))
        if compare:
            # low1, up1, low2, up2 = bootstrap_metric_2(target, pred, mean_per_class_recall, Acc)
            # print("置信结果:", low1, up1, low2, up2)
            return {"balanced_acc": mean_per_class_recall, "acc": acc}, target.cpu(), pred.cpu()
        else:
            # low1, up1 = bootstrap_metric(target, pred, mean_per_class_recall)
            # print("置信结果:", low1, up1)
            return {"balanced_acc": mean_per_class_recall}, target.cpu(), pred.cpu()


def evaluate1(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, save_clf=None,
              load_clfs=[]):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names

    templates: list of str
        templates to use for zero-shot classification

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    print("Evaluating1...")
    cnt = 0
    total = 0
    gts = []
    predicts = []
    text_label_list = [templates[0].format(c=classname) for classname in classnames]
    # print(text_label_list)
    # text = tokenizer(text_label_list).cuda()
    model = model.to(device)
    # pred, target = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            with autocast(device_type=device):
                text_features = get_text_embeddings(model, tokenizer, text_label_list, device).float()
                image_features = get_image_embeddings(model, images).float()
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predict_labels = torch.argmax(text_probs, dim=-1)
            cnt += (predict_labels == labels).sum().item()
            gts.extend(labels.cpu().numpy())
            predicts.extend(predict_labels.cpu().numpy())

            if total % 100 == 0:
                print(cnt / total)
    # 召回率的平均值
    mean_per_class_recall = balanced_accuracy_score(gts, predicts)
    # low1, up1, low2, up2 = bootstrap_metric_2(gts, predicts, mean_per_class_recall, Acc)
    # print("置信结果:", low1, up1, low2, up2)
    print(cnt / total)
    # with open('output.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # 写入表头
    #     writer.writerow(['Ground Truth', 'Prediction'])
    #     # 逐行写入数据
    #     for gt, pred in zip(gts, predicts):
    #         writer.writerow([gt, pred])
    return {"balanced_acc": mean_per_class_recall, "acc": cnt / total}, gts, predicts
