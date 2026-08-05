# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""
import copy
import json
import os
import tempfile

from fsdet.data import MetadataCatalog, DatasetCatalog
from .register_coco import register_coco_instances
from .meta_coco import register_meta_coco
from .lvis import register_lvis_instances
from .meta_lvis import register_meta_lvis
from .pascal_voc import register_pascal_voc
from .meta_pascal_voc import register_meta_pascal_voc
from .builtin_meta import _get_builtin_metadata
from ...structures import BoxMode
from fsdetection import load_fs_dataset

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}


def register_all_coco(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    # register meta datasets
    METASPLITS = [
        ("coco_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_train_freq": ("coco/train2017", "lvis/lvis_v0.5_train_freq.json"),
        "lvis_v0.5_train_common": ("coco/train2017", "lvis/lvis_v0.5_train_common.json"),
        "lvis_v0.5_train_rare": ("coco/train2017", "lvis/lvis_v0.5_train_rare.json"),
        "lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/val2017", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/test2017", "lvis/lvis_v0.5_image_info_test.json"),
    },
}


def register_all_lvis(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_lvis_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    # register meta datasets
    METASPLITS = [
        ("lvis_v0.5_train_shots", "coco/train2017", "lvissplit/lvis_shots.json"),
        ("lvis_v0.5_train_rare_novel", "coco/train2017", "lvis/lvis_v0.5_train_rare.json"),
        ("lvis_v0.5_val_novel", "coco/val2017", "lvis/lvis_v0.5_val.json"),
    ]

    for name, image_root, json_file in METASPLITS:
        dataset_name = "lvis_v0.5_fewshot" if "novel" in name else "lvis_v0.5"
        register_meta_lvis(
            name,
            _get_builtin_metadata(dataset_name),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    # register meta datasets
    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(100):
                        seed = '' if seed == 0 else '_seed{}'.format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed)
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid)
                        keepclasses = "base_novel_{}".format(sid) \
                            if prefix == 'all' else "novel{}".format(sid)
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid))

    # print(METASPLITS)
    # ('voc_2007_trainval_all1_1shot', 'VOC2007', 'all_1shot_split_1_trainval', 'base_novel_1', 1)
    # ('voc_2007_trainval_novel3_10shot', 'VOC2007', 'novel_10shot_split_3_trainval', 'novel3', 3)
    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_pascal_voc(name,
                                 _get_builtin_metadata("pascal_voc_fewshot"),
                                 os.path.join(root, dirname), split, year,
                                 keepclasses, sid)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


    # register fine-tune dataset with more base (3ploidy)
    WITH_MORE_BASE = []
    ploidy = 3
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(100):
                        seed = '' if seed == 0 else '_seed{}'.format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}_{}ploidy".format(
                            year, prefix, sid, shot, seed, ploidy)
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid)
                        keepclasses = "base_novel_{}".format(sid) \
                            if prefix == 'all' else "novel{}".format(sid)
                        WITH_MORE_BASE.append(
                            (name, dirname, img_file, keepclasses, sid))

    for name, dirname, split, keepclasses, sid in WITH_MORE_BASE:
        year = 2007 if "2007" in name else 2012
        register_meta_pascal_voc(name,
                                 _get_builtin_metadata("pascal_voc_fewshot"),
                                 os.path.join(root, dirname), split, year,
                                 keepclasses, sid)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

# ==== Datasets for Cross-Domain from Huggingface ===========
def hf_to_detectron2(dataset, split="train"):
    records = []

    for idx, sample in enumerate(dataset):
        width, height = sample["image"].size

        record = {
            "file_name": None,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": [],
        }

        for bbox, cat_id in zip(
                sample["objects"]["bbox"],
                sample["objects"]["category"]
        ):
            record["annotations"].append({
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat_id,
            })

        record["image"] = sample["image"]
        records.append(record)

    return records

def hf_to_coco_dict(dataset, categories):
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    images_dict = {}

    ann_id = 1

    for img_id, sample in enumerate(dataset):
        width, height = sample["image"].size

        coco["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": f"{img_id}.jpg",
        })
        images_dict[img_id] = sample["image"]

        for bbox, cat_id in zip(
            sample["objects"]["bbox"],
            sample["objects"]["category"]
        ):
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            ann_id += 1

    return coco, images_dict

def write_temp_coco(coco_dict):
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode='w', delete=False
    )
    json.dump(coco_dict, tmp)
    tmp.close()
    return tmp.name

def register_hf_data():
    seed = os.getenv("REPEAT_ID", 2026)
    dataset_name = os.getenv("DATASET")

    dataset = load_fs_dataset(f"/lustre/fsn1/projects/rech/mvq/ubc18yy/datasets/{dataset_name}")
    og_dataset = copy.deepcopy(dataset["train"])
    classes = dataset["train"].features["objects"]["category"].feature.names

    id2label = dict(enumerate(classes))
    categories = [{"id": i, "name": name} for i, name in id2label.items()]

    coco_dict, images_dict_test = hf_to_coco_dict(dataset["test"], categories=categories)
    coco_path = write_temp_coco(coco_dict)

    register_coco_instances(f"{dataset_name}_test", {}, coco_path, image_root=".")
    DatasetCatalog.register(f"{dataset_name}_test_images", lambda: images_dict_test)
    MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=classes, evaluator_type="coco")
    del coco_dict

    coco_dict, images_dict_val = hf_to_coco_dict(dataset["validation"], categories=categories)
    coco_path = write_temp_coco(coco_dict)

    register_coco_instances(f"{dataset_name}_val", {}, coco_path, image_root=".")
    DatasetCatalog.register(f"{dataset_name}_val_images", lambda: images_dict_val)
    MetadataCatalog.get(f"{dataset_name}_val").set(thing_classes=classes, evaluator_type="coco")
    del coco_dict

    name = f"{dataset_name}_1shot"
    dataset["train"].sampling(shots=1, seed=int(seed))
    records_1shot = hf_to_detectron2(dataset["train"])
    DatasetCatalog.register(name, lambda: records_1shot)
    MetadataCatalog.get(name).set(thing_classes=classes)
    dataset["train"] = copy.deepcopy(og_dataset)

    name = f"{dataset_name}_5shot"
    dataset["train"].sampling(shots=5, seed=int(seed))
    records_5shot = hf_to_detectron2(dataset["train"])
    DatasetCatalog.register(name, lambda: records_5shot)
    MetadataCatalog.get(name).set(thing_classes=classes)
    dataset["train"] = copy.deepcopy(og_dataset)

    name = f"{dataset_name}_10shot"
    dataset["train"].sampling(shots=10, seed=int(seed))
    records_10shot = hf_to_detectron2(dataset["train"])
    DatasetCatalog.register(name, lambda: records_10shot)
    MetadataCatalog.get(name).set(thing_classes=classes)
    dataset["train"] = copy.deepcopy(og_dataset)

    del og_dataset


# Register them all under "./datasets"
register_all_coco()
register_all_lvis()
register_all_pascal_voc()
register_hf_data()
