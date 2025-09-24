<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 16px;">
  <img src="pictures/logo.png" alt="LEXam Logo" width="120" style="border: none;">
  <div style="text-align: left;">
    <h1 style="margin: 0;">LEXam: Benchmarking Legal Reasoning on 340 Law Exams</h1>
    <p style="margin: 6px 0 0;">A diverse, rigorous evaluation suite for legal AI from Swiss, EU, and international law examinations.</p>
  </div>
</div>

### This repository provides code for evaluating LLMs on ***LEXam***. [[Dataset]](https://github.com/y-fn/LEXam/tree/main/data)

## 🚀🔄 Reproducing Paper results or Evaluating your own LLM

### Environment Preparation
```shell
git clone https://github.com/LEXam-Benchmark/LEXam
cd LEXam
conda create -n lexam python=3.11
conda activate lexam
cd lighteval
pip install -e .[dev]
cd ..
pip install -r requirements.txt

# Set API keys for inference and evaluation.
# For the current example code, OpenAI key is mandatory, as we use GPT-4o as expert-verified grader to illustrate the pipeline.
EXPORT OPENAI_API_KEY="xxx"
EXPORT TOGETHER_API_KEY="xxx"
EXPORT DEEPSEEK_API_KEY="xxx"
EXPORT ANTHROPIC_API_KEY="xxx"
EXPORT GEMINI_API_KEY="xxx"
```

### Evaluating Non-Reasoning LLMs with [[Huggingface lighteval]](https://huggingface.co/docs/lighteval/index)
Huggingface lighteval provides the advantage of uniformly evaluating LLMs from different endpoints -- local vLLM, OpenAI, Anthropic, TogetherAI, Gemini ...

Together-AI, OpenAI, Gemini, and other API-based LLMs can be evaluated by:
```shell
MODEL="openai/gpt-4o-mini-2024-07-18" 

# Evaluating GPT-4o-mini on LEXam Open Question subset.
python -m lighteval endpoint litellm "${MODEL}" "community|lexamoq_open_question|0|0" --custom-tasks lighteval/community_tasks/lexam_oq_evals.py --output-dir outputs_oq --save-details --use-chat-template

# Evaluating GPT-4o-mini on LEXam Multiple-Choice Question subset.
python -m lighteval endpoint litellm "${MODEL}" "community|lexammcq_mcq_4_choices|0|0" --custom-tasks lighteval/community_tasks/lexam_mcq_evals.py --output-dir outputs_mcq --save-details --use-chat-template
```
- `MODEL`: the target LLM you are evaluating, e.g., `openai/gpt-4.1`, `together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- `--output-dir`: evaluation results will be saved to `--output-dir`.
- `--save-details`: details including prompts, LLM responses, LLM judges, and other evaluation metrics will be saved in `details`.

Local inference using vLLM:
```shell
MODEL="meta-llama/Llama-3.1-8B-Instruct" 
export HF_HOME="xxx"
export HUGGINGFACE_TOKEN="xxx"
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Evaluating GPT-4o-mini on LEXam Open Question subset.
python -m lighteval vllm "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16" "community|lexamoq_open_question|0|0" --custom-tasks lighteval/community_tasks/lexam_oq_evals.py --output-dir outputs_oq --save-details --use-chat-template

# Evaluating GPT-4o-mini on LEXam Multiple-Choice Question subset.
python -m lighteval vllm "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16" "community|lexammcq_mcq_4_choices|0|0" --custom-tasks lighteval/community_tasks/lexam_mcq_evals.py --output-dir outputs_mcq --save-details --use-chat-template
```

### Evaluating Reasoning LLMs with LiteLLM directly.
Reasoning LLMs generate both a <think> scratch pad and the final answer after </think>. To only evaluate the answer, we do not use lighteval for reasoning LLMs.
```shell
MODEL="deepseek-reasoner"
python litellm_eval.py --input_file data/open_questions_test.xlsx --cache_name r1 --llm $MODEL --output_file lexam_oq_${MODEL}.csv --batch_size 2 --task_type open_quesitons
python litellm_eval.py --input_file data/MCQs_test.xlsx --cache_name r1 --llm $MODEL --output_file lexam_mcq_${MODEL}.csv --batch_size 2 --answer_field gold --task_type mcq_letters
```
- `MODEL` can be set to any model included in `MODEL_DICT` of `litellm_eval.py`, e.g., `o1`, `o3-mini`, `qwq-32b`.
- `--output_file`: DeepSeek-R1's answer to open/MC questions will be at `lexam_oq_deepseek-reasoner.csv` and `lexam_mcq_deepseek-reasoner.csv`
- `--task_type`: chose from ['mcq_letters', 'mcq_numbers', 'open_questions']. mcq_letters and _numbers differ by using ABCD or 1234 as choice labels.

Then evaluate the answers using our expert-verified LLM judge. This script will print the Mean and bootstrapped Variance of open question performance.
```shell
MODEL="deepseek-reasoner"
python customized_judge_async.py --input_file lexam_oq_${MODEL}.csv --output_file lexam_oq_${MODEL}_graded.csv --async_call --cache_name gpt4o --llm gpt-4o
```
- `--input_file`: Grade DeepSeek-R1's answer to open questions. Grading results at `lexam_oq_deepseek-reasoner_graded.csv`


Finally evaluate the accuracy of MCQs. This script will print accuracy and bootstrapped variance. No LLM call is involved in this script.
```shell
MODEL="deepseek-reasoner"
INPUT_FILE="lexam_mcq_${MODEL}.csv"
python evaluation.py --input_file $INPUT_FILE --response_field ${MODEL}_answer --task_type mcq_letters
```
## Licenses

- The **Code** in this repository is licensed under the [Apache License 2.0](LICENSE).
- The **Data** in this repository is licensed under the [Creative Commons Attribution 4.0 International License](LICENSE_DATA).

