"""Console script for clip_benchmark."""
import argparse
import sys
import random
import json

import open_clip
import torch
import csv
from copy import copy
import os
from itertools import product

from PIL import Image
from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, get_dataset_default_task, \
    get_dataset_collection_from_file
from clip_benchmark.metrics import zeroshot_classification, zeroshot_retrieval, linear_probe, image_retrieval
from clip_benchmark.model_collection import get_model_collection_from_file, model_collection
from clip_benchmark.models import load_clip, MODEL_TYPES
import torch.nn as nn
import torchvision
import numpy as np
# from sympy.printing.tests.test_tensorflow import tf
from transformers import CLIPTokenizerFast
from transformers.models import clip



# def convert_to_list(data):
#     if isinstance(data, list):
#         return data
#     elif isinstance(data, (np.ndarray, torch.Tensor, tf.Tensor)):
#         if isinstance(data, np.ndarray):
#             return data.tolist()
#         elif isinstance(data, torch.Tensor):
#             return data.tolist()
#         elif isinstance(data, tf.Tensor):
#             return data.numpy().tolist()
#     else:
#         return [data]


def get_parser_args():
    # import os
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument("--corrupt", type=int, default=0)
    parser_eval.add_argument("--single_template", type=int, default=0)
    parser_eval.add_argument("--corrupt_all", type=int, default=0)
    parser_eval.add_argument("--corrupt_level", type=str, default="a")
    parser_eval.add_argument("--corruption_types", type=str, default=None, help="brightness|contrast|saturation")
    parser_eval.add_argument("--augmentation", type=int, default=0)
    parser_eval.add_argument("--augmentation_types", type=str, default=None,
                             help="stain normalisation|all augmentations|rotation_flipping")
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+",
                             help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file where each line is a dataset name")
    parser_eval.add_argument('--dataset_root', default="root", type=str,
                             help="dataset root folder where the datasets are downloaded. Can be in the form of a template depending on dataset name, e.g., --dataset_root='datasets/{dataset}'. This is useful if you evaluate on multiple datasets.")
    parser_eval.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--model', type=str, nargs="+", default=["ViT-B-32-quickgelu"],
                             help="Model architecture to use from OpenCLIP")
    parser_eval.add_argument('--pretrained', type=str, nargs="+", default=["laion400m_e32"],
                             help="Model checkpoint name to use from OpenCLIP")
    parser_eval.add_argument('--pretrained_model', type=str, default="", nargs="+",
                             help="Pre-trained model(s) to use. Can be the full model name where `model` and `pretrained` are comma separated (e.g., --pretrained_model='ViT-B-32-quickgelu,laion400m_e32'), a model collection name ('openai' or 'openclip_base' or 'openclip_multilingual' or 'openclip_all'), or path of a text file where each line is a model fullname where model and pretrained are comma separated (e.g., ViT-B-32-quickgelu,laion400m_e32). --model and --pretrained are ignored if --pretrained_model is used.")
    parser_eval.add_argument('--task', type=str, default="auto",
                             choices=["zeroshot_classification", "zeroshot_retrieval",
                                      "linear_probe", "captioning",
                                      "image_caption_selection", "auto",
                                      "image_retrieval", "pathvqa"],
                             help="Task to evaluate on. With --task=auto, the task is automatically inferred from the dataset.")
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=True,
                             help="whether to use mixed precision")
    parser_eval.add_argument('--num_workers', default=4, type=int)
    parser_eval.add_argument('--recall_k', default=[5], type=int,
                             help="for retrieval, select the k for Recall@K metric. ", nargs="+", )
    parser_eval.add_argument('--fewshot_k', default=-1, type=int,
                             help="for linear probe, how many shots. -1 = whole dataset.")
    parser_eval.add_argument('--fewshot_epochs', default=10, type=int, help="for linear probe, how many epochs.")
    parser_eval.add_argument('--fewshot_lr', default=0.1, type=float,
                             help="for linear probe, what is the learning rate.")
    parser_eval.add_argument("--skip_load", action="store_true",
                             help="for linear probes, when everything is cached, no need to load model.")
    parser_eval.add_argument("--ms_aug", action="store_true",
                             help="whether or not use multi-scale augmentation for MUSK")

    parser_eval.add_argument("--distributed", action="store_true", help="evaluation in parallel")
    parser_eval.add_argument('--seed', default=0, type=int, help="random seed.")
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument('--batch_size_eval', default=256, type=int)
    parser_eval.add_argument('--model_cache_dir', default=None, type=str,
                             help="directory to where downloaded models are cached")
    parser_eval.add_argument('--feature_root', default="features", type=str,
                             help="feature root folder where the features are stored.")
    parser_eval.add_argument('--annotation_file', default="", type=str,
                             help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser_eval.add_argument('--custom_classname_file', default=None, type=str,
                             help="use custom json file with classnames for each dataset, where keys are dataset names and values are list of classnames.")
    parser_eval.add_argument('--custom_template_file', default=None, type=str,
                             help="use custom json file with prompts for each dataset, where keys are dataset names and values are list of prompts. For instance, to use CuPL prompts, use --custom_template_file='cupl_prompts.json'")

    parser_eval.add_argument('--language', default="en", type=str, nargs="+",
                             help="language(s) of classname and prompts to use for zeroshot classification.")
    parser_eval.add_argument('--output', default="result.json", type=str,
                             help="output file where to dump the metrics. Can be in form of a template, e.g., --output='{dataset}_{pretrained}_{model}_{language}_{task}.json'")
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false", help="suppress verbose messages")
    parser_eval.add_argument('--save_clf', default=None, type=str,
                             help="optionally save the classification layer output by the text tower")
    parser_eval.add_argument('--load_clfs', nargs='+', default=[], type=str,
                             help="optionally load and average mutliple layers output by text towers.")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true",
                             help="whether to skip an evaluation if the output file exists.")
    parser_eval.add_argument('--model_type', default="open_clip", type=str, choices=MODEL_TYPES, help="clip model type")
    parser_eval.add_argument('--wds_cache_dir', default=None, type=str,
                             help="optional cache directory for webdataset only")
    parser_eval.set_defaults(which='eval')

    parser_build = subparsers.add_parser('build', help='Build CSV from evaluations')
    parser_build.add_argument('files', type=str, nargs="+", help="path(s) of JSON result files")
    parser_build.add_argument('--output', type=str, default="benchmark.csv", help="CSV output file")
    parser_build.set_defaults(which='build')

    args = parser.parse_args()
    return parser, args


def main():
    # torch.cuda.empty_cache()
    parser, base = get_parser_args()
    print("-------------in-main--------------")
    if not hasattr(base, "which"):
        parser.print_help()
        return
    if base.which == "eval":
        main_eval(base)
    elif base.which == "build":
        main_build(base)


def main_build(base):
    # Build a benchmark single CSV file from a set of evaluations (JSON files)
    rows = []
    fieldnames = set()
    for path in base.files:
        data = json.load(open(path))
        row = {}
        row.update(data["metrics"])
        row.update(data)
        del row["metrics"]
        row['model_fullname'] = row['model'] + ' ' + row['pretrained']
        for field in row.keys():
            fieldnames.add(field)
        rows.append(row)
    with open(base.output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main_eval(base):
    # Get list of pre-trained models to evaluate
    print("-------------in main_eval-------------")
    pretrained_model = _as_list(base.pretrained_model)
    if pretrained_model:
        models = []
        for name in pretrained_model:
            if os.path.isfile(name):
                # if path, read file, each line is a pre-trained model
                models.extend(get_model_collection_from_file(name))
            elif name in model_collection:
                # if part of `model_collection`, retrieve from it
                models.extend(model_collection[name])
            else:
                # if not, assume it is in the form of `model,pretrained`
                model, pretrained = name.split(',')
                models.append((model, pretrained))
    else:
        models = list(product(base.model, base.pretrained))

    # Ge list of datasets to evaluate on
    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            # If path, read file, each line is a dataset name
            datasets.extend(get_dataset_collection_from_file(name))
        else:
            # if not, assume it is simply the name of the dataset
            datasets.append(name)

    # Get list of languages to evaluate on
    languages = _as_list(base.language)

    if base.verbose:
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        print(f"Languages: {languages}")
    runs = product(models, datasets, languages)
    if base.distributed:
        local_rank, rank, world_size = world_info_from_env()
        runs = list(runs)
        # randomize runs so that runs are balanced across gpus
        random.seed(base.seed)
        random.shuffle(runs)
        runs = [r for i, r in enumerate(runs) if i % world_size == rank]
    for (model, pretrained), (dataset), (language) in runs:
        # We iterative over all possible model/dataset/languages
        args = copy(base)
        args.model = model
        args.pretrained = pretrained
        args.dataset = dataset
        args.language = language
        run(args)


def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l


# seed everything
def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)  # torch < 1.8
    torch.use_deterministic_algorithms(True, warn_only=True)  # torch >= 1.8

    # disable TF32 on Ampere (A6000) and Ada (L40) GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run(args, transforms=None):
    """Console script for clip_benchmark."""
    # config
    CORRUPTION_CONFIG = {
        'brightness': {
            'range': (0.0, 2.5),  # 实际参数范围
            'convert': lambda s: 1 + s * 1.5,
            'presets': {
                'mild': 0.3,  # 1.45x
                'moderate': 0.6,  # 1.9x
                'severe': 1.0  # 2.5x
            }
        },
        'contrast': {
            'range': (0.3, 1.0),
            'convert': lambda s: 1 - s * 0.7,
            'presets': {
                'mild': 0.2,  # 0.86x
                'moderate': 0.5,  # 0.65x
                'severe': 1.0  # 0.3x
            }
        },
        'gaussian_blur': {
            'range': (0, 5.0),  # 像素半径
            'convert': lambda s: s * 5,
            'presets': {
                'mild': 0.2,  # 1px
                'moderate': 0.5,  # 2.5px
                'severe': 1.0  # 5px
            }
        },
        'resolution': {
            'range': (0.3, 1.0),  # 缩放比例
            'convert': lambda s: 1 - s * 0.7,
            'presets': {
                'mild': 0.2,  # 0.86x
                'moderate': 0.5,  # 0.65x
                'severe': 1.0  # 0.3x
            }
        },
        'saturation': {
            'range': (0.2, 1.0),
            'convert': lambda s: 1 - s * 0.8,
            'presets': {
                'mild': 0.25,  # 0.8x
                'moderate': 0.6,  # 0.52x
                'severe': 1.0  # 0.2x
            }
        },
        'hue_shift': {
            'range': (0, 50),  # 色调偏移量
            'convert': lambda s: int(s * 50),
            'presets': {
                'mild': 0.2,  # +10
                'moderate': 0.5,  # +25
                'severe': 1.0  # +50
            }
        },
        'markup': {
            'range': (0, 1),  # 存在概率
            'convert': lambda s: s,
            'presets': {
                'mild': 0.3,  # 30%概率
                'moderate': 0.7,  # 70%概率
                'severe': 1.0  # 100%概率
            }
        }
    }
    print("----------------in_run----------------")
    import torch
    if torch.cuda.is_available():
        if args.distributed:
            local_rank, rank, world_size = world_info_from_env()
            device = 'cuda:%d' % local_rank
            torch.cuda.set_device(device)
        else:
            device = "cuda"
        args.device = device
    else:
        args.device = "cpu"
    # set seed.
    fix_seed(args.seed)
    task = args.task
    if args.dataset.startswith("wds/"):
        dataset_name = args.dataset.replace("wds/", "", 1)
    else:
        dataset_name = args.dataset
    if task == "auto":
        task = get_dataset_default_task(dataset_name)
    pretrained_slug = os.path.basename(args.pretrained) if os.path.isfile(args.pretrained) else args.pretrained
    pretrained_slug_full_path = args.pretrained.replace('/', '_') if os.path.isfile(
        args.pretrained) else args.pretrained
    dataset_slug = dataset_name.replace('/', '_')
    output = args.output.format(
        model=args.model,
        pretrained=pretrained_slug,
        pretrained_full_path=pretrained_slug_full_path,
        task=task,
        dataset=dataset_slug,
        language=args.language
    )
    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return
    if args.verbose:
        print(f"Running '{task}' on '{dataset_name}' with the model '{args.pretrained}' on language '{args.language}'")

    data_root = args.dataset_root
    dataset_path = {
        "skin": f"{data_root}/skincancer",
        "pannuke": f"{data_root}/pannuke",
        "unitopatho": f"{data_root}/unitopatho/unitopath-public",
        "unitopatho_retrieval": f"{data_root}/unitopatho/unitopath-public",  # image2image retrieval
        "pathmmu_retrieval": f"{data_root}/PathMMU",  # cross-modal retrieval
        "bookset_retrieval": f"{data_root}/books_set",
        "pubmedset_retrieval": f"{data_root}/pubmed_set",
        "pvqa": f"{data_root}/pvqa",
        "LC25000": f"{data_root}/LC25000",
        "LC25000_lung": f"{data_root}/LC25000",
        "PatchCamelyon": f"{data_root}/PatchCamelyon/pcamv1-20250218T085908Z-002",
        "CRC100K": f"{data_root}/CRC100K",
        "CRC100K_norm": f"{data_root}/CRC100K",
        "BACH": f"{data_root}/BACH",
        "Osteo": f"{data_root}/Osteo",
        "WSSS4LUAD": f"{data_root}/WSSS4LUAD",
        "SICAPv2": f"{data_root}/SICAPv2",
        "BRACS_Rol": f"{data_root}/BRACS",
        "BRACS_Rol_seven": f"{data_root}/BRACS",
        "GCHTID":f"{data_root}/GCHTID",
        "DIA":f"{data_root}/Desmoplastic_infantile_astrocytoma",
        "PXA":f"{data_root}/Pleomorphic_xanthoastrocytoma",
        "SGCA":f"{data_root}/Subependymal_giant_cell_astrocytoma",
        "PMA":f"{data_root}/Pilomyxoid_astrocytoma",
        "PA":f"{data_root}/Pilocytic_astrocytoma"
    }

    dataset_root = dataset_path[dataset_name]
    # --------------------model-load--------------------------------------
    print(args.model)
    if 'musk' in args.model.lower():
        model_type = 'musk'
    elif 'conchv1_5' in args.model.lower():
        model_type = 'conchv1_5'
    elif 'conch' in args.model.lower():
        model_type = 'conch'
    elif 'pathgenclip-l' in args.model.lower():
        model_type = 'pathgenclip-l'
    elif 'pathgenclip' in args.model.lower():
        model_type = 'pathgenclip'
    elif 'plip' in args.model.lower():
        model_type = 'plip'
    elif 'quilt' in args.model.lower():
        model_type = 'quilt'
    elif 'biomedclip' in args.model.lower():
        model_type = 'biomedclip'
    elif 'pathclip' in args.model.lower():
        model_type = 'pathclip'
    elif 'pmcclip' in args.model.lower():
        model_type = 'pmcclip'
    elif 'pubmedclip' in args.model.lower():
        model_type = 'pubmedclip'
    elif 'clip' in args.model.lower():
        model_type = 'clip'
    elif 'uni-2-h' in args.model.lower():
        model_type = 'UNI-2-h'
    elif 'uni' in args.model.lower():
        model_type = 'UNI'
    elif 'virchow2' in args.model.lower():
        model_type = 'Virchow2'
    elif 'prov-gigapath' in args.model.lower():
        model_type = 'Prov-GigaPath'
    else:
        model_type = 'clip'
        # raise NotImplementedError
    from huggingface_hub import login
    #  >>>> load models >>>> #
    if args.skip_load:
        model, transform, collate_fn, dataloader = None, None, None, None
        tokenizer = None
        dataset = None
        train_dataloader = None

    else:
        if model_type == 'clip':
            print("----------{}------------".format(model_type))
            model, transform, tokenizer = load_clip(
                model_type=args.model_type,
                model_name=args.model,
                pretrained=args.pretrained,
                cache_dir=args.model_cache_dir,
                device=args.device
            )
            print(tokenizer("Tumor"))
            print(tokenizer("tumor"))
            model.eval()
        elif model_type == 'conchv1_5':
            from benchmarks.clip_benchmark.models.conchv1_5.conchv1_5 import create_model_from_pretrained
            model, transform = create_model_from_pretrained("./conch_v1_5_official/pytorch_model_vision.bin")
            model.eval()
        elif model_type == 'musk':
            from transformers import XLMRobertaTokenizer
            from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
            from timm.models import create_model
            from musk import modeling
            from musk import utils as mutils
            import huggingface_hub
            print("----------{}------------".format(model_type))
            # tokenizer_path = args.pretrained.replace("musk.pth", "tokenizer.spm")
            local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
            local_path = os.path.join('/root/autodl-tmp/MUSK/benchmarks/clip_benchmark/local_musk/model/')
            # print(local_dir)
            hub_name = args.pretrained.split(":")[1]
            print(hub_name)
            # 下载tokenizer.spm
            # huggingface_hub.hf_hub_download(
            #     hub_name,
            #     filename="tokenizer.spm",
            #     local_dir=local_dir,
            #     force_download=True
            # )
            # tokenizer_path = os.path.join(local_dir, "tokenizer.spm")
            tokenizer_path = os.path.join(local_path, "tokenizer.spm")
            tokenizer = XLMRobertaTokenizer(tokenizer_path)

            print(tokenizer("Tumor"))
            print(tokenizer("tumor"))

            img_size = 384 if '384' in args.model else 224

            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size, interpolation=3, antialias=True),
                torchvision.transforms.CenterCrop((img_size, img_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            ])

            model = create_model(args.model)
            # print(model)
            # load model weight
            # print(args.pretrained)
            mutils.load_model_and_may_interpolate(args.pretrained, model, 'model|module', '', local_path)
            model.eval()

        elif model_type == 'conch':
            from clip_benchmark.models.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
            tokenizer = get_tokenizer()
            print("----------{}------------".format(model_type))
            model, transform = create_model_from_pretrained(
                "conch_ViT-B-16",
                checkpoint_path=args.pretrained
            )

            print(tokenizer("Tumor"))
            print(tokenizer("tumor"))
            model.eval()
            # model = model.to(device)
            # allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为 MB
            # print(f"当前已分配的显存: {allocated_memory:.2f} MB")
        elif model_type == 'pathgenclip':
            print("----------{}------------".format(model_type))
            # from config import ratio, epoch, mode, path, pretrained, model_type,
            #                                                                         force_quick_gelu=True
            model, _, transform = open_clip.create_model_and_transforms('ViT-B-16', pretrained=args.pretrained,
                                                                        force_quick_gelu=True)
            model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
            # tokenizer = open_clip.get_tokenizer('ViT-B-32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
        elif model_type == 'pathgenclip-l':
            print("----------{}------------".format(model_type))
            # from config import ratio, epoch, mode, path, pretrained, model_type
            model, _, transform = open_clip.create_model_and_transforms('ViT-L-14', pretrained=args.pretrained)
            tokenizer = open_clip.get_tokenizer('ViT-L-14')
            # model, _, transform = open_clip.create_model_and_transforms('ViT-B-16', pretrained=args.pretrained,
            #                                                             force_quick_gelu=True)
            model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
            # tokenizer = open_clip.get_tokenizer('ViT-B-32')
            # tokenizer = open_clip.get_tokenizer('ViT-B-16')
        elif model_type == 'pmcclip':
            print("----------{}------------".format(model_type))
            model, preprocess_train, transform = open_clip.create_model_and_transforms(args.pretrained)
            tokenizer = open_clip.get_tokenizer(args.pretrained)
            model.eval()
            # image = transform(Image.open("example.png")).unsqueeze(0)
            # text = tokenizer(["An H&E image of tumor patch", "An H&E image of normal patch"])
        elif model_type == 'quilt':
            print("----------{}------------".format(model_type))
            model, preprocess_train, transform = open_clip.create_model_and_transforms(args.pretrained)
            tokenizer = open_clip.get_tokenizer(args.pretrained)
            print(tokenizer("Tumor"))
            print(tokenizer("tumor"))
            model.eval()
        elif model_type == 'biomedclip':
            print("----------{}------------".format(model_type))
            model, preprocess_train, transform = open_clip.create_model_and_transforms(args.pretrained)
            tokenizer = open_clip.get_tokenizer(args.pretrained)
            model.eval()
        elif model_type == 'plip':
            from PIL import Image
            from transformers import CLIPProcessor, CLIPModel
            print("----------{}------------".format(model_type))
            model = CLIPModel.from_pretrained(args.pretrained)
            transform = CLIPProcessor.from_pretrained(args.pretrained)
            tokenizer = CLIPTokenizerFast.from_pretrained(args.pretrained)

            print(tokenizer("Tumor"))
            print(tokenizer("tumor"))

            model.eval()
        elif model_type == 'pathclip':
            print("----------{}------------".format(model_type))
            model, _, transform = open_clip.create_model_and_transforms('ViT-B-16',
                                                                        pretrained=args.pretrained,
                                                                        cache_dir='/mnt/Xsky/syx/model/open_clip',
                                                                        force_quick_gelu=True)
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model = model.cuda()
            model.eval()
        elif model_type == 'pubmedclip':
            print("----------{}------------".format(model_type))
            model_v = args.model.split('-')[1]
            if model_v == 'ViT':
                model_v = args.model.split('-')[1] + '-' + args.model.split('-')[2] + '-' + args.model.split('-')[3]
            model, _, transform = open_clip.create_model_and_transforms(model_v, pretrained=args.pretrained)
            model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
            tokenizer = open_clip.get_tokenizer(model_v)
            print(tokenizer("Tumor"))
            print(tokenizer("tumor"))
        # foundation model
        elif model_type == 'UNI-2-h':
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            from huggingface_hub import login
            print("----------{}------------".format(model_type))
            # login with your User Access Token, found at https://huggingface.co/settings/tokens
            # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
            timm_kwargs = {
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667 * 2,
                'num_classes': 0,
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True
            }
            model = timm.create_model(args.pretrained, pretrained=True, **timm_kwargs)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            model.eval()
        elif model_type == 'UNI':
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            from huggingface_hub import login
            print("----------{}------------".format(model_type))
            # login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

            # pretrained=True needed to load UNI weights (and download weights for the first time)
            # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
            model = timm.create_model(args.pretrained, pretrained=True, init_values=1e-5, dynamic_img_size=True)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            model.eval()
        elif model_type == 'Virchow2':
            import timm
            import torch
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            from timm.layers import SwiGLUPacked
            from PIL import Image

            # need to specify MLP layer and activation function for proper init
            print("----------{}------------".format(model_type))
            model = timm.create_model(args.pretrained, pretrained=True, mlp_layer=SwiGLUPacked,
                                      act_layer=torch.nn.SiLU)
            model = model.eval()

            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        elif model_type == 'Prov-GigaPath':
            import timm
            from torchvision import transforms
            # os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp'
            print("----------{}------------".format(model_type))
            # online_download
            # model = timm.create_model(args.pretrained, pretrained=True)
            # local
            model = timm.create_model(args.pretrained, pretrained=True, pretrained_cfg_overlay=dict(
                file='/root/autodl-tmp/MUSK/benchmarks/clip_benchmark/models/Prov-GigaPath/pytorch_model.bin'))

            transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            model.eval()



        # elif model_type == 'PMC-CLIP':
        # elif model_type == 'PathCLIP':
        # elif model_type == 'PubmedCLIP':
        else:
            raise NotImplementedError
        # ----------------------build_dataset----------------------------------------------
        if args.corrupt_all:
            print('---------------Corrupting all----------------------------')
            corrupt_type_all = ['hue_shift', 'brightness', 'contrast', 'saturation', 'gaussian_blur']
            corrupt_level_all = ['mild', 'moderate', 'severe']
            for type in corrupt_type_all:
                corruption_type = []
                corruption_type.append(type)
                for level in corrupt_level_all:
                    torch.cuda.empty_cache()
                    corrupt_level = []
                    corrupt_level.append(CORRUPTION_CONFIG[type]['presets'][level])
                    print(corruption_type, corrupt_level)
                    dataset = build_dataset(
                        dataset_name=args.dataset,
                        root=dataset_root,
                        transform=transform,
                        split=args.split,
                        annotation_file=args.annotation_file,
                        download=True,
                        language=args.language,
                        task=task,
                        custom_template_file=args.custom_template_file,
                        custom_classname_file=args.custom_classname_file,
                        wds_cache_dir=args.wds_cache_dir,
                        corrupt=args.corrupt,
                        corrupt_level=corrupt_level,
                        corruption_types=corruption_type
                    )

                    collate_fn = get_dataset_collate_fn(args.dataset)
                    if args.verbose:
                        try:
                            print(f"Dataset size: {len(dataset)}")
                        except TypeError:
                            print("IterableDataset has no len()")
                        print(f"Dataset split: {args.split}")
                        if hasattr(dataset, "classes") and dataset.classes:
                            try:
                                print(f"Dataset classes: {dataset.classes}")
                                print(f"Dataset number of classes: {len(dataset.classes)}")
                            except AttributeError:
                                print("Dataset has no classes.")

                    if args.dataset.startswith("wds/"):
                        dataloader = torch.utils.data.DataLoader(
                            dataset.batched(args.batch_size), batch_size=None,
                            shuffle=False, num_workers=args.num_workers,
                        )
                    else:
                        dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn
                        )
                        # 检查当前任务是否为零样本分类任务
                    if task == "zeroshot_classification":
                        # 尝试从数据集中获取零样本分类模板
                        # 如果数据集对象有 "templates" 属性，则使用该属性的值作为零样本模板
                        # 否则，将零样本模板设为 None
                        if args.single_template:
                            zeroshot_templates = dataset.single_template if hasattr(dataset, "templates") else None
                        else:
                            zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
                        # 如果命令行参数中指定了需要详细输出信息
                        if args.verbose:
                            # 打印零样本分类模板的信息
                            print(f"Zero-shot templates: {zeroshot_templates}")
                        # 尝试从数据集中获取分类类别名称
                        # 如果数据集对象有 "classes" 属性，则使用该属性的值作为分类类别名称
                        # 否则，将分类类别名称设为 None
                        classnames = dataset.classes if hasattr(dataset, "classes") else None
                        # 断言零样本模板和分类类别名称都不为 None
                        # 如果不满足条件，会抛出 AssertionError 异常，并提示数据集不支持分类任务
                        assert (
                                zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
                        # 调用 zeroshot_classification 模块中的 evaluate 函数进行零样本分类评估
                        if args.single_template:
                            metrics, target, pred = zeroshot_classification.evaluate1(
                                # 传入用于评估的模型
                                model,
                                # 传入数据加载器，用于批量加载数据
                                dataloader,
                                # 传入分词器，用于对文本进行分词处理
                                tokenizer,
                                # 传入分类类别名称
                                classnames,
                                # 传入零样本分类模板
                                zeroshot_templates,
                                # 指定模型运行的设备，如 "cpu" 或 "cuda"
                                device=args.device,
                                # 是否使用自动混合精度（AMP）进行训练或评估
                                amp=args.amp,
                                # 是否需要详细输出信息
                                verbose=args.verbose,
                                # 是否保存分类器
                                save_clf=args.save_clf,
                                # 是否加载已有的分类器
                                load_clfs=args.load_clfs,
                            )
                        else:
                            metrics, target, pred = zeroshot_classification.evaluate(
                                # 传入用于评估的模型
                                model,
                                # 传入数据加载器，用于批量加载数据
                                dataloader,
                                # 传入分词器，用于对文本进行分词处理
                                tokenizer,
                                # 传入分类类别名称
                                classnames,
                                # 传入零样本分类模板
                                zeroshot_templates,
                                # 指定模型运行的设备，如 "cpu" 或 "cuda"
                                device=args.device,
                                # 是否使用自动混合精度（AMP）进行训练或评估
                                amp=args.amp,
                                # 是否需要详细输出信息
                                verbose=args.verbose,
                                # 是否保存分类器
                                save_clf=args.save_clf,
                                # 是否加载已有的分类器
                                load_clfs=args.load_clfs,
                            )
                    elif task == "zeroshot_retrieval":  # vision-language multi-modal retrieval

                        metrics = zeroshot_retrieval.evaluate(
                            model,
                            dataloader,
                            tokenizer,
                            recall_k_list=args.recall_k,
                            device=args.device,
                            amp=args.amp
                        )

                    elif task == "image_retrieval":  # image retrieval
                        metrics = image_retrieval.evaluate(
                            model,
                            dataloader,
                            recall_k_list=args.recall_k,
                            device=args.device,
                            amp=args.amp,
                            model_id=model_type
                        )

                    elif task == "linear_probe":
                        # we also need the train split for linear probing.
                        train_dataset = build_dataset(
                            dataset_name=args.dataset,
                            root=dataset_root,
                            transform=transform,
                            split='train',
                            annotation_file=args.annotation_file,
                            download=True,
                        )
                        train_dataloader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, pin_memory=True,
                        )

                        metrics = linear_probe.evaluate(
                            model,
                            train_dataloader,
                            dataloader,
                            args.fewshot_k,
                            args.batch_size_eval,
                            args.num_workers,
                            args.fewshot_lr,
                            args.fewshot_epochs,
                            (args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
                            args.seed,
                            args.feature_root,
                            device=args.device,
                            amp=args.amp,
                            verbose=args.verbose,
                            ms_aug=args.ms_aug
                        )

                    else:
                        raise ValueError("Unsupported task: {}".format(task))

                    dump = {
                        "dataset": args.dataset,
                        "model": args.model,
                        "pretrained": args.pretrained,
                        "task": task,
                        "metrics": metrics,
                        "language": args.language,
                    }
                    os.makedirs("results", exist_ok=True)
                    if args.corrupt:
                        output = os.path.dirname(output)
                        basename = "corrupt_" + type + "_" + level + ".json"
                        output = os.path.join(output, basename)
                    if args.verbose:
                        print(f"Dump results to: {output}")
                    with open(output, "a+") as f:
                        json.dump(dump, f)
                        f.write('\n')
            return 0
        else:
            # corruption_type = args.corruption_types
            # corrupt_level = args.corrupt_level
            corruption_type = []
            corrupt_level = []
            if args.corrupt:
                corruption_type = args.corruption_types.split('|')
                corrupt_level1 = args.corrupt_level.split('|')
                corrupt_level = []
                for level, type in zip(corrupt_level1, corruption_type):
                    corrupt_level.append(CORRUPTION_CONFIG[type]['presets'][level])
                # print(corrupt_level)
            # print('corruption_type')
            dataset = build_dataset(
                dataset_name=args.dataset,
                root=dataset_root,
                transform=transform,
                split=args.split,
                annotation_file=args.annotation_file,
                download=True,
                language=args.language,
                task=task,
                custom_template_file=args.custom_template_file,
                custom_classname_file=args.custom_classname_file,
                wds_cache_dir=args.wds_cache_dir,
                corrupt=args.corrupt,
                corrupt_level=corrupt_level,
                corruption_types=corruption_type,
                augmentation=args.augmentation,
                augmentation_types=args.augmentation_types
            )
            allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为 MB
            print(f"当前已分配的显存: {allocated_memory:.2f} MB")

            collate_fn = get_dataset_collate_fn(args.dataset)
            if args.verbose:
                try:
                    print(f"Dataset size: {len(dataset)}")
                except TypeError:
                    print("IterableDataset has no len()")
                print(f"Dataset split: {args.split}")
                if hasattr(dataset, "classes") and dataset.classes:
                    try:
                        print(f"Dataset classes: {dataset.classes}")
                        print(f"Dataset number of classes: {len(dataset.classes)}")
                    except AttributeError:
                        print("Dataset has no classes.")

            if args.dataset.startswith("wds/"):
                dataloader = torch.utils.data.DataLoader(
                    dataset.batched(args.batch_size), batch_size=None,
                    shuffle=False, num_workers=args.num_workers,
                )
            else:
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_fn
                )
    # -----------------------------TASK------------------------------------------------
    # 检查当前任务是否为零样本分类任务
    if task == "zeroshot_classification":
        # 尝试从数据集中获取零样本分类模板
        # 如果数据集对象有 "templates" 属性，则使用该属性的值作为零样本模板
        # 否则，将零样本模板设为 None
        if args.single_template:
            zeroshot_templates = dataset.single_template if hasattr(dataset, "templates") else None
        else:
            zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
        # 如果命令行参数中指定了需要详细输出信息
        if args.verbose:
            # 打印零样本分类模板的信息
            print(f"Zero-shot templates: {zeroshot_templates}")
        # 尝试从数据集中获取分类类别名称
        # 如果数据集对象有 "classes" 属性，则使用该属性的值作为分类类别名称
        # 否则，将分类类别名称设为 None
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        print(classnames)
        # 断言零样本模板和分类类别名称都不为 None
        # 如果不满足条件，会抛出 AssertionError 异常，并提示数据集不支持分类任务
        assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
        # 调用 zeroshot_classification 模块中的 evaluate 函数进行零样本分类评估
        if args.single_template:
            metrics, target, pred = zeroshot_classification.evaluate1(
                # 传入用于评估的模型
                model,
                # 传入数据加载器，用于批量加载数据
                dataloader,
                # 传入分词器，用于对文本进行分词处理
                tokenizer,
                # 传入分类类别名称
                classnames,
                # 传入零样本分类模板
                zeroshot_templates,
                # 指定模型运行的设备，如 "cpu" 或 "cuda"
                device=args.device,
                # 是否使用自动混合精度（AMP）进行训练或评估
                amp=args.amp,
                # 是否需要详细输出信息
                verbose=args.verbose,
                # 是否保存分类器
                save_clf=args.save_clf,
                # 是否加载已有的分类器
                load_clfs=args.load_clfs,
            )
        else:
            # allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2  # 转换为 MB
            # print(f"当前已分配的显存: {allocated_memory:.2f} MB")
            metrics, target, pred = zeroshot_classification.evaluate(
                # 传入用于评估的模型
                model,
                # 传入数据加载器，用于批量加载数据
                dataloader,
                # 传入分词器，用于对文本进行分词处理
                tokenizer,
                # 传入分类类别名称
                classnames,
                # 传入零样本分类模板
                zeroshot_templates,
                # 指定模型运行的设备，如 "cpu" 或 "cuda"
                device=args.device,
                # 是否使用自动混合精度（AMP）进行训练或评估
                amp=args.amp,
                # 是否需要详细输出信息
                verbose=args.verbose,
                # 是否保存分类器
                save_clf=args.save_clf,
                # 是否加载已有的分类器
                load_clfs=args.load_clfs,
            )
    elif task == "zeroshot_retrieval":  # vision-language multi-modal retrieval

        metrics, text_embs, image_embs, texts_image_index = zeroshot_retrieval.evaluate(
            model,
            dataloader,
            tokenizer,
            recall_k_list=args.recall_k,
            device=args.device,
            amp=args.amp
        )

    elif task == "image_retrieval":  # image retrieval
        metrics, images_emb, image_labels = image_retrieval.evaluate(
            model,
            dataloader,
            recall_k_list=args.recall_k,
            device=args.device,
            amp=args.amp,
            model_id=model_type
        )

    elif task == "linear_probe":
        # we also need the train split for linear probing.
        train_dataset = build_dataset(
            dataset_name=args.dataset,
            root=dataset_root,
            transform=transform,
            split='train',
            annotation_file=args.annotation_file,
            download=True,
            augmentation=args.augmentation,
            augmentation_types=args.augmentation_types,
            task=task
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True,
        )
        print("Datasize: ", len(train_dataset))
        # print(args.device)
        metrics = linear_probe.evaluate(
            model,
            train_dataloader,
            dataloader,
            args.fewshot_k,
            args.batch_size_eval,
            args.num_workers,
            args.fewshot_lr,
            args.fewshot_epochs,
            (args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
            args.seed,
            args.feature_root,
            device=args.device,
            amp=args.amp,
            verbose=args.verbose,
            ms_aug=args.ms_aug,
            augmentation=args.augmentation,
            augmentation_types=args.augmentation_types
        )

    else:
        raise ValueError("Unsupported task: {}".format(task))
    # --------------------------------------------Save_result-------------------------------
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
    }
    os.makedirs("results", exist_ok=True)
    basename = os.path.basename(output)
    basename_csv = os.path.basename(output).replace('json', 'csv')
    dirname = os.path.dirname(output)
    output_csv = os.path.join(dirname, args.dataset, basename_csv)
    if args.dataset == "LC25000":
        output = os.path.join(dirname, args.dataset, "colon", basename)
        output_csv = os.path.join(dirname, args.dataset, "colon", basename_csv)
    elif args.dataset == "LC25000_lung":
        output = os.path.join(dirname, "LC25000", "lung", basename)
        output_csv = os.path.join(dirname, "LC25000", "lung", basename_csv)
    elif args.dataset == "CRC100K_norm":
        output = os.path.join(dirname, args.dataset,'norm', basename)
    else:
        output = os.path.join(dirname, args.dataset, basename)
    if args.augmentation:
        output = os.path.join(os.path.dirname(output), args.augmentation_types, os.path.basename(output))

    if args.corrupt:
        output = os.path.dirname(output)
        basename = "corrupt_" + args.corruption_types + "_" + args.corrupt_level + ".json"
        basename_csv = "corrupt_" + args.corruption_types + "_" + args.corrupt_level + ".csv"
        output = os.path.join(output, basename)
        output_csv = os.path.join(output, basename_csv)
    if args.verbose:
        print(f"Dump results to: {output}")
    # if not os.path.exists(os.path.basename(os.path.dirname(output))):
    #     os.mkdir(os.path.basename(os.path.dirname(output)))
    # 以追加模式打开文件
    if not os.path.exists(os.path.dirname(output_csv)):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if task == 'zeroshot_retrieval':
        # print(positive_pairs)
        # print(scores)
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 检查文件是否为空，如果为空则写入表头
            if csvfile.tell() == 0:
                writer.writerow(['model', 'text_embs', 'image_embs', 'texts_image_index'])
            # 逐行写入数据
            # print(target.device)
            if isinstance(text_embs, torch.Tensor):
                text_embs = text_embs.tolist()
            if isinstance(image_embs, torch.Tensor):
                image_embs = image_embs.tolist()
            if isinstance(texts_image_index, torch.Tensor):
                texts_image_index = texts_image_index.tolist()
            writer.writerow([args.model, text_embs, image_embs, texts_image_index])
    elif task == 'zeroshot_classification':
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 检查文件是否为空，如果为空则写入表头
            if csvfile.tell() == 0:
                writer.writerow(['model', 'Ground Truth', 'pred'])
            # 逐行写入数据
            # print(target.device)
            if isinstance(target, torch.Tensor):
                target = target.tolist()
            if isinstance(pred, torch.Tensor):
                pred = pred.tolist()
            for gt, p in zip(target, pred):
                writer.writerow([args.model, gt, p])
    elif task == 'image_retrieval':
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 检查文件是否为空，如果为空则写入表头
            if csvfile.tell() == 0:
                writer.writerow(['model', 'image_emb', 'image_labels'])
            # 逐行写入数据
            # print(target.device)
            if isinstance(images_emb, torch.Tensor):
                images_emb = images_emb.tolist()
            if isinstance(image_labels, torch.Tensor):
                image_labels = image_labels.tolist()
            writer.writerow([args.model, images_emb, image_labels])


    print(output)
    output_dir = os.path.dirname(output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # exist_ok=True避免目录已存在时出错
    with open(output, "a+") as f:

        # print(dump)
        json.dump(dump, f)
        f.write('\n')
    # print("close")
    # f.close()
    # torch.cuda.empty_cache()

    return 0


def world_info_from_env():
    # from openclip
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


if __name__ == "__main__":
    print("--main---")
    sys.exit(main())  # pragma: no cover
