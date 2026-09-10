import json
import os
from typing import Optional, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .msr_gist_dataset import MSRGistDataset
from .xplus_gemma import GemmaXPlusGenerator, DEFAULT_PROMPT_TEMPLATE


class MSRGistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        tokenizer_name: str = "bert-base-uncased",
        batch_size: int = 4,
        max_length: int = 128,
        num_workers: int = 0,
        y_key: str = "summary",
        include_xt: bool = False,

        # x+ on-the-fly settings
        make_xplus: bool = True,
        gemma_model_dir: Optional[str] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        deterministic_xplus: bool = False,  # optional future: seed
    ):
        super().__init__()
        self.train_path = train_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.y_key = y_key
        self.include_xt = include_xt

        self.make_xplus = make_xplus
        self.gemma_model_dir = gemma_model_dir
        self.prompt_template = prompt_template
        self.deterministic_xplus = deterministic_xplus

        self.tokenizer = None
        self.xplus_generator = None

        self.train_dataset = None

    def prepare_data(self):
        # tokenizer downloads only if not present; should already be ok
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # IMPORTANT: for Gemma generation do NOT use num_workers>0 unless you redesign multiprocessing
        if self.make_xplus and self.num_workers != 0:
            raise ValueError("For on-the-fly Gemma x+, set num_workers=0 (single process).")

        # Load Gemma generator once
        if self.make_xplus:
            if not self.gemma_model_dir:
                raise ValueError("make_xplus=True but gemma_model_dir not provided.")
            self.xplus_generator = GemmaXPlusGenerator(model_dir=self.gemma_model_dir)

        self.train_dataset = MSRGistDataset(
            data_path=self.train_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            y_key=self.y_key,
            include_xt=self.include_xt,
            xplus_generator=self.xplus_generator if self.make_xplus else None,
            prompt_template=self.prompt_template,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
