# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
This module contains task configurations and prompt functions for evaluating
LLM models on the LEXam dataset.

Author: Jingwei Ni
"""
import importlib.metadata as importlib_metadata
import random
from typing import Callable, Literal
import time
import logging
import re
import sys
import os
import ast
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset
import torch
from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.utils.imports import is_litellm_available, is_openai_available, is_vllm_available
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.extended.mix_eval.main import process_judge_response_freeform_gpt
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from huggingface_hub import HfApi
from aenum import extend_enum
from lighteval.metrics.metrics import Metrics


logger = logging.getLogger(__name__)

MCQ_PROMPT_KEY = "letters"
CHOICE_LIST = ['A', 'B', 'C', 'D']
MATCH_CHOICE_REGEX = r"###([ABCD])###"
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
# Try to optimize CUDA operations
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
    # Enable TF32 for faster matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True
    # Enable tensor cores if available
    if torch.cuda.get_device_capability()[0] >= 7:
        # This will speed up GPU inference, e.g., for COMET and BLEURT
        torch.set_float32_matmul_precision("medium")


MCQ_PROMPT = {
    "letters": """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., A, B, C, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., A, B, C, etc.) with a brief justification for why it best fits the legal analysis.

Format your final answer as follows:
 Correct Answer: ###C### 

Question:
 {question}

Answer:""",
    "numbers": """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., 1, 2, 3, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., 1, 2, 3, etc.) with a brief justification for why it best fits the legal analysis.

Format your final answer as follows:
 Correct Answer: ###3### 

Question:
 {question}

Answer:""",
}


def prompt_fn(sample, task_name: str = None):
    question = sample["Question"]
    if isinstance(sample["Choices"], list):
        choice_list = sample["Choices"]
    else:
        choice_list = ast.literal_eval(sample["Choices"])
    for letter, c in zip(CHOICE_LIST, choice_list):
        question += f'\n{letter}. {c}'
    query = MCQ_PROMPT[MCQ_PROMPT_KEY].format(course_name=sample["Course"], question=question)
    return Doc(
        task_name=task_name,
        query=query,
        choices=CHOICE_LIST,
        gold_index=sample["Gold"],
        instruction="",
    )


def sample_level_acc(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    response = predictions[0]
    matches = re.findall(MATCH_CHOICE_REGEX, response)
    if matches:
        matched_letter = matches[-1]
    else:
        return {"mc_acc": 0}
    return {"mc_acc": matched_letter == formatted_doc.choices[formatted_doc.gold_index]}


def agg_function(items):
    score = sum(items) / len(items)
    return score * 100


mc_acc_metric = SampleLevelMetricGrouping(
    metric_name=["mc_acc"],
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=sample_level_acc,
    corpus_level_fn={
        "mc_acc": agg_function,
    },
)


class LEXamMCQTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=f'{name}_{hf_subset}',
            hf_subset=hf_subset,
            prompt_function=prompt_fn,
            hf_repo="JingweiNi/LEXam",
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            metric=[mc_acc_metric],
            suite=["community"],
            generation_size=4096,
            stop_sequence=['</s>'],
            trust_dataset=True,
        )


TASKS_TABLE = [LEXamMCQTask(name="lexammcq", hf_subset="mcq_4_choices")]

if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
