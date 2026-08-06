# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in FsDet.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import copy
import json
import os
import tempfile

import fsdet.utils.comm as comm
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.config import get_cfg, set_global_cfg
from fsdet.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from fsdet.data.dataset_mapper import DatasetMapperHuggingFace
from fsdet.data.datasets import register_coco_instances
from fsdet.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from fsdet.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from fsdetection import load_fs_dataset

from fsdet.structures import BoxMode


# from fsdet.data.dataset_mapper import AlbumentationMapper

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

        # record["image"] = sample["image"]
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

    name = f"{dataset_name}_train"
    records = hf_to_detectron2(dataset["train"])
    DatasetCatalog.register(name, lambda: records)
    MetadataCatalog.get(name).set(thing_classes=classes)
    dataset["train"] = copy.deepcopy(og_dataset)

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

    del dataset
    return og_dataset

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = register_hf_data()
        mapper = DatasetMapperHuggingFace(cfg, is_train=True, hf_dataset=dataset)
        # if cfg.INPUT.USE_ALBUMENTATIONS:
        #     mapper = AlbumentationMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
