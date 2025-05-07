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
LLM models on the LExBench dataset.

Author: Jingwei Ni
"""
import importlib.metadata as importlib_metadata
from typing import Callable, Literal
import time
import logging
import re
import sys
import os
import statistics
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


logger = logging.getLogger(__name__)

USE_MINI = False
JUDGE_PROMPT_KEY = "20250324"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Try to optimize CUDA operations
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
    # Enable TF32 for faster matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True
    # Enable tensor cores if available
    if torch.cuda.get_device_capability()[0] >= 7:
        # This will speed up GPU inference, e.g., for COMET and BLEURT
        torch.set_float32_matmul_precision("medium")

# CUSTOM METRICS

QA_PROMPT = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
Assume Swiss law applies unless specifically mentioned; if the course context justifies, address legal issues beyond Swiss law as well.
Use precise legal language and formal "Sie" when answering.
Do NOT state any disclaimer or refer to the need for external legal advice.
Do NOT request the user to consult laws or to research on their own.
Offer focused legal analyses and individualized advice.
Speak directly and authoritatively without mentioning that your response is merely for general information.
Incorporate Swiss-specific legal terminology.
If you have discovered relevant legal considerations (Erwägungen), respond with a concise, clear legal analysis.
Cite only from your identified considerations.
Always cite the specific legal provision, explicitly indicating paragraphs (Abs.), numbers (Ziff.), or letters (lit.) where available (e.g., “'Art. 74 Abs. 2 Ziff. 2 OR”, “Art. 336 lit. a StGB”). Avoid general references (such as 'Art. 3 ZGB') without mentioning the specific paragraph, number, or letter, if applicable.
If no relevant considerations are found, explicitly state that no pertinent information is available.
If you do have reliable sources, share practical guidance or insights from them.
Respond in the same language as the question.
If the question specifically requests a short answer, provide a concise response.
If the prompt asks you to analyze a specific case provided in the exam, but the text or details of that case have not been provided in the prompt, explicitly flag that the required case material is missing.

Question:
{question}

Answer:"""

JUDGE_SYSTEM = {
    "basic": "Act as a Judge specializing in the evaluation of university exams of Swiss law schools. Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.",
    "20250310": "Act as a Judge specializing in the evaluation of Swiss law schools exams. Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.",
    "20250324": "Act as a Judge specializing in the evaluation of Swiss law schools exams. Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.",
}

JUDGE_USER = {
    "basic": """You will be provided with a response to a law school exam question (question) and a reference answer (reference). Your task is to evaluate how accurately and completely the response (response) meets the reference answer, with a focus on accuracy, completeness, and legal reasoning.

In some cases, the reference answer may include only keywords or factual elements to be examined, along with (+) or (–) indicating whether each element should be affirmed or denied. In other instances, (-/+) may indicate that arguments in both directions are acceptable if they are legally sound. The reference answer may also contain citations (e.g., from books or articles), which the response is not required to replicate.

After reviewing the response:
Briefly explain your reasoning regarding how the response conforms to or deviates from the reference answer. Assign a final correctness score on a scale from 0.0 to 1.0 (in increments of 0.1). This score should reflect the extent to which the response satisfies the reference answer, where 1.0 indicates complete fulfillment (100%) and lower scores reflect proportionate shortfalls (e.g. 0.5 indicates 50% fulfillment). The correctness score must strictly follow this format: "[[score]]", e.g., "The correctness score: [[0.5]]". Additionally, provide neutral, constructive feedback and corrections in the style of a university professor.
""",
    "20250310": """Goal:
    Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.

Return format:
    After reviewing the response:
    1. Explanation: Briefly explain your reasoning regarding how the response conforms to or deviates from the reference answer. 
    2. Correctness score: Assign a final correctness score on a scale from 0.0 to 1.0 (in increments of 0.1). This score should reflect the extent to which the response satisfies the reference answer, where 
        - 1.0 = complete fulfillment (100%) 
        - lower scores reflect proportionate shortfalls (e.g. 0.5 = 50% fulfillment). 
        - Strictly follow the format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". 
    3. Constructive feedback: Additionally, provide neutral, constructive feedback and corrections in the style of a university professor.

Warnings:
    - In some cases, the reference answer may include only keywords or factual elements to be examined, along with (+), (-) or (+/-). Respect these indications when determining correctness:
        - (+) means the element must be affirmed.
        - (–) means the element must be denied.
        - (-/+) indicates that arguments in either direction are acceptable if legally sound.
    - Deviations or additional elements not found in the reference answer should generally be penalized unless you are certain they are legally correct and relevant. Assume the reference answer includes all information necessary for a perfect response.
    - The reference answer may contain citations (e.g., from books or law review articles), which the response does not need to replicate. However, statutes should be cited precisely, specifying Abs., Ziff., or lit. whenever applicable.
    - If the reference answer includes separate sub-points, use these for proportional scoring guidance (e.g., addressing 2 out of 4 sub-points correctly equals approximately a 0.5 score).

Context:
    You will be provided with a response to a law school exam question (labeled: question) and a reference answer (labeled: reference). 
""",
    "20250324": """Goal:
Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.

Context:
You will be provided with a response (labeled: Model's Answer) to a law school exam question (labeled: Question) and a reference answer (labeled: Reference Answer). 

Return format:
    After reviewing the response:
    1. Explanation: Briefly explain your reasoning regarding how the response conforms to or deviates from the reference answer. 
    2. Constructive feedback: Additionally, provide neutral, constructive feedback and corrections in the style of a university professor.
    3. Correctness score: Assign a final correctness score on a scale from 0.0 to 1.0 (in increments of 0.1). This score should reflect the extent to which the response satisfies the reference answer, where 
        - 1.0 = complete fulfillment (100%) 
        - lower scores reflect proportionate shortfalls (e.g. 0.5 = 50% fulfillment). 
	    - strictly follow the format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". 

Warnings:
    - In some cases, the reference answer may include only keywords or factual elements to be examined, along with (+), (-) or (+/-). Respect these indications when determining correctness:
        - (+) means the element must be affirmed.
        - (–) means the element must be denied.
        - (-/+) indicates that arguments in either direction are acceptable if legally sound.
    - Deviations or additional elements not found in the reference answer should generally be penalized unless you are certain they are legally correct and relevant. Assume the reference answer includes all information necessary for a perfect response.
    - The reference answer may contain citations (e.g., from books or law review articles), which the response does not need to replicate. However, statutes should be cited precisely, specifying Abs., Ziff., or lit. whenever applicable.
    - If the reference answer includes separate sub-points, use these for proportional scoring guidance (e.g., addressing 2 out of 4 sub-points correctly equals approximately a 0.5 score).
""",
}

FEW_SHOT = {
    "no": """""",
}


class JudgeLExBench(JudgeLLM):

    def compute(
        self,
        sample_ids: list[str],
        responses: list,
        formatted_docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        logger.info(f"Judging {len(formatted_docs)} samples with {self.short_judge_name}...")
        questions = [formatted_doc.specific["question_fact"] for formatted_doc in formatted_docs]
        options = [formatted_doc.choices for formatted_doc in formatted_docs]
        golds = [formatted_doc.get_golds()[0] for formatted_doc in formatted_docs]
        predictions = [response[0].result for response in responses]

        scores, _, judgements = self.judge.evaluate_answer_batch(questions, predictions, options, golds)
        # Exclude the messages (user prompt) because they are too long
        return [
            {
                self.short_judge_name: score * 100,
                f"{self.short_judge_name}_judgment": judgment,
            }
            for score, judgment in zip(scores, judgements)
        ]


def process_judge_response_grade_gpt(x):
    search = re.search(r"\[\[(\d.\d)\]\]", x)
    answer = float(search.group(1) if search else 0)
    if answer > 1 or answer < 0:
        answer = 0
    return answer


def get_lexbench_law_exam_judge(
    judge_model_name: str = "openai/gpt-4o-mini-2024-07-18" if USE_MINI else "openai/gpt-4o-2024-11-20",
    short_judge_name: str = "exam_judge_gpt-4o-mini" if USE_MINI else "exam_judge_gpt-4o",
    backend: str = "litellm",
    system_style: str = JUDGE_PROMPT_KEY,  # basic or 20250310
    few_shot_style: str = "no",
):
    def lexbench_law_exam_judge(question, options, answer, gold):
        system_prompt = JUDGE_SYSTEM[system_style]
        user = JUDGE_USER[system_style]
        few_shot_examples = FEW_SHOT[few_shot_style]
        instruction = f"""Judge the below case, give the brief reasoning process and the final grade.


Question:
```{question}```

Reference Answer:
```{gold}```

Model's Answer:
```{answer}```

Your Judgment:
"""
        user_prompt = user + few_shot_examples + instruction
        # logger.info(user_prompt)

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    return SampleLevelMetricGrouping(
        metric_name=[short_judge_name],
        higher_is_better={short_judge_name: True},
        category=MetricCategory.LLM_AS_JUDGE,
        use_case=MetricUseCase.NONE,
        sample_level_fn=JudgeLExBench(
            judge_model_name=judge_model_name,
            template=lexbench_law_exam_judge,
            process_judge_response=process_judge_response_grade_gpt,
            judge_backend=backend,
            short_judge_name=short_judge_name,
        ).compute,
        corpus_level_fn={short_judge_name: statistics.mean},
    )


# EVALS WITH SUBSET
def prompt_fn(line: dict, task_name: str = None):
    """
    Prompt function for law exam question answering.
    """
    custom_query = QA_PROMPT.format(course_name=line['Course'], question=line['Question'])

    return Doc(
        task_name=task_name,
        query=custom_query,
        choices=[str(line['Answer'])],
        gold_index=0,
        specific={
            "question_fact": line['Question'],
        },
    )


@dataclass
class DatasetConfig:
    name: str
    hf_repo: str
    subsets: list[str]


JUDGE_MODELS = {"gpt-4o-mini": "openai/gpt-4o-mini-2024-07-18"} if USE_MINI else {"gpt-4o": "openai/gpt-4o-2024-11-20",}


API_METRICS = ["exam_judge_gpt-4o-mini"] if USE_MINI else ["exam_judge_gpt-4o"]

JUDGE_METRICS = [
    f"exam_judge_{judge_model}-{system_style}-{few_shot_style}".replace("-", "_")
    for judge_model in JUDGE_MODELS
    for system_style in [JUDGE_PROMPT_KEY]
    for few_shot_style in ["no"]
]

metrics_to_evaluate = ["judge"]

METRICS_TO_USE = []
if metrics_to_evaluate == ["debug"]:
    METRICS_TO_USE = ["bleu"]
elif "api" in metrics_to_evaluate:
    METRICS_TO_USE += API_METRICS
elif "judge" in metrics_to_evaluate:
    METRICS_TO_USE += JUDGE_METRICS
else:
    METRICS_TO_USE = API_METRICS
logger.info(f"Available metrics: {METRICS_TO_USE}")

METRICS = {}


def init_llm_judge_metric(metric_name: str):
    if metric_name == "exam_judge_gpt-4o":
        METRICS["exam_judge_gpt-4o"] = get_lexbench_law_exam_judge(
            judge_model_name="openai/gpt-4o-mini-2024-07-18" if USE_MINI else "openai/gpt-4o-2024-11-20",
            short_judge_name="exam_judge_gpt-4o-mini" if USE_MINI else "exam_judge_gpt-4o",
        )

    # Check all the judge metric combinations
    for judge_model in JUDGE_MODELS:
        for system_style in [JUDGE_PROMPT_KEY]:
            for few_shot_style in ["no"]:
                short_judge_name = f"exam_judge_{judge_model}-{system_style}-{few_shot_style}"
                judge_metric_name = short_judge_name.replace("-", "_")
                if metric_name == judge_metric_name:
                    METRICS[metric_name] = get_lexbench_law_exam_judge(
                        judge_model_name=JUDGE_MODELS[judge_model],
                        short_judge_name=short_judge_name,
                        system_style=system_style,
                        few_shot_style=few_shot_style,
                    )
                    break


def init_metric(metric_name: str):
    # Only load the metric once
    if metric_name in METRICS:
        logger.debug(f"Metric {metric_name} already initialized")
        return

    # ===== LLM Judge metrics =====
    init_llm_judge_metric(metric_name)


# Additionally we could consider adding the following open source judge models:
# flowaicom/Flow-Judge-v0.1, prometheus-eval/prometheus-7b-v2.0
# However, these are only fine-tuned on English data and we need multilingual support.


def get_metrics(METRICS_TO_USE):
    metrics = []
    for metric in METRICS_TO_USE:
        init_metric(metric)
        metrics.append(METRICS[metric])
    return metrics


class ExamQATask(LightevalTaskConfig):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        hf_subset: str,
    ):
        super().__init__(
            name=f"{dataset_config.name}_{hf_subset}",
            suite=["community"],
            prompt_function=prompt_fn,
            hf_repo=dataset_config.hf_repo,
            hf_subset=hf_subset,
            hf_filter=None,
            hf_avail_splits=["dev", "test"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            metric=get_metrics(METRICS_TO_USE),
            generation_size=4096,
            stop_sequence=['</s>'],
            trust_dataset=True,
            # Remove the target language in the beginning if it exists: e.g., FR: {translation}
            # Is only applied to the generative metrics, but also there seems not to be invoked, maybe not passed through?
            # output_regex=f"(?:{target_lang.upper()}:\s*?)?(.*)",
        )


# STORE YOUR EVALS
LExBenchQA = DatasetConfig(
    name="lexbenchoq",
    hf_repo="JingweiNi/LExBench",
    subsets=['open_question'],
)

# list of all the subsets to use for this eval
DATASETS = [
    LExBenchQA,
]

TASKS_TABLE = [
    ExamQATask(
        dataset_config=dataset,
        hf_subset=subset,
    )
    for dataset in DATASETS
    for subset in dataset.subsets
]


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))

