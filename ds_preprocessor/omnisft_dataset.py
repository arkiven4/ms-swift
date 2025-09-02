import datasets
from datasets import GeneratorBasedBuilder, DatasetInfo, BuilderConfig, Split, Features, Value, Audio, Image, Sequence
import os
import tarfile
import pandas as pd
from datasets.utils.logging import tqdm
from qwen_omni_utils import process_mm_info
import io
import soundfile as sf
import numpy as np
import ast
from PIL import Image as ImagePIL
import math
import glob

# datasets.logging.set_verbosity_info()

_DESCRIPTION = """\
This is a sample dataset.
"""


def numpy_to_wav_bytes(audio_array: np.ndarray, sr: int) -> bytes:
    """
    Convert a NumPy array to WAV bytes in memory.
    """
    buf = io.BytesIO()
    sf.write(buf, audio_array, sr, format='WAV')
    return buf.getvalue()


def crop_and_convertJPG(path_file, max_size=1024):
    array_PIL = ImagePIL.open(path_file)

    width, height = array_PIL.size
    max_dim = max(width, height)

    if max_dim > max_size:
        scale = max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        array_PIL = array_PIL.resize(
            (new_width, new_height), resample=ImagePIL.Resampling.LANCZOS)

    return array_PIL


class MyAudioDatasetConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OmniGRPODataset(GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = MyAudioDatasetConfig
    BUILDER_CONFIGS = [
        MyAudioDatasetConfig(
            name="default",
            description="OmniGRPO default config",
            version=datasets.Version("1.0.0"),
            data_dir="/home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets"
        )
    ]
    DEFAULT_CONFIG_NAME = "default"
    print("------------------------ SFT Datasets Processor ---------------------------")

    def _info(self):
        return DatasetInfo(
            description=_DESCRIPTION,
            features=Features({
                "messages": [{
                    "content": Value("string"),
                    "role": Value("string")
                }],
                "solution": Value("string"),
                "audios": Sequence(Value("binary")),
                "images": [{"bytes": Value("binary"), "path": Value("null")}],
            })
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir

        all_shards_files = []
        for split in ["train"]:
            csv_file = os.path.join(data_dir, f"{split}_balancediseaseaugment.csv")
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(csv_file):
                continue

            tar_parts = sorted(
                [f for f in os.listdir(split_dir) if f.endswith(".tar")])
            num_files_in_dir = len([
                f for f in os.listdir(split_dir)
                if os.path.isfile(os.path.join(split_dir, f))
            ])

            # merge/extract tar parts if needed
            # if os.path.exists(csv_file) and len(tar_parts) > 0:
            #     tar_path = os.path.join(split_dir, f"{split}.tar")
            #     with open(tar_path, "wb") as out:
            #         for part in tqdm(tar_parts, desc=f"Merging {split} parts"):
            #             with open(os.path.join(split_dir, part), "rb") as f:
            #                 out.write(f.read())
            #     with tarfile.open(tar_path) as tar:
            #         tar.extractall(path=split_dir)
            #     os.remove(tar_path)


            os.makedirs(os.path.join(data_dir, "shards_metadata"), exist_ok=True)
            num_shards = 16
            df = pd.read_csv(csv_file)
            shard_size = math.ceil(len(df) / num_shards)
            for i in range(num_shards):
                start = i * shard_size
                end = start + shard_size
                shard_path = os.path.join(data_dir, "shards_metadata", f"{split}_{i}.csv")
                df.iloc[start:end].to_csv(shard_path, index=False)
                all_shards_files.append(shard_path)

        # single split only
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"csv_files": all_shards_files, "data_dir": data_dir},
            )
        ]

    def _generate_examples(self, csv_files, data_dir):
        idx = 0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            desired_order = ["identifier", "messages",
                             "solution", "audio_file_name", "image_file_name"]
            df = df[[c for c in desired_order if c in df.columns]]

            for _, row in df.iterrows():
                conversation = ast.literal_eval(row.get("messages"))
                example = {
                    "messages": None,
                    "solution": row.get("solution"),
                    "audios": None,
                    "images": None
                }

                for ele in conversation[1]['content']:
                    if ele["type"] == "audio":
                        if "audio" in ele or "audio_url" in ele:
                            ele["audio"] = os.path.join(
                                data_dir, row.get("audio_file_name"))
                    if ele["type"] == "image":
                        if "image" in ele:
                            ele["image"] = crop_and_convertJPG(
                                os.path.join(data_dir, row.get("image_file_name")))

                #del conversation[0]
                audios, images, _ = process_mm_info(conversation, use_audio_in_video=False)
                example["messages"] = [
                    {
                        "role": item["role"],
                        "content": next(
                            (ele["text"] for ele in item["content"]
                             if ele["type"] not in ("audio", "image")),
                            ""
                        )
                    }
                    for item in conversation
                ]

                if images != None:
                    example["images"] = images
                if audios != None:
                    example["audios"] = [numpy_to_wav_bytes(audios[0], 16000)]

                yield idx, example
                idx += 1
