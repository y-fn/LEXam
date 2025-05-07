<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 16px;">
  <img src="pictures/logo.png" alt="LExBench Logo" width="120" style="border: none;">
  <div style="text-align: left;">
    <h1 style="margin: 0;">LExBench: Benchmark Legal Reasoning with Legal Exams</h1>
    <p style="margin: 6px 0 0;">A diverse, rigorous evaluation suite for legal AI from Swiss Legal Exams.</p>
  </div>
</div>

### This Repo provides code for evaluating LLMs on LExBench. [[Huggingface Dataset]](https://huggingface.co/datasets/JingweiNi/LExBench) [[Paper]]()

## 🔥 News
- [2025/05] Release of the first version of paper, where we evaluate 20+ representative SoTA LLMs with evaluations stricly verified by legal experts.

## 🚀🔄 Reproducing Paper results or Evaluating your own LLM


### Environment Preparation
```shell
git clone https://github.com/EdisonNi-hku/LExBench
cd LExBench
conda create -n lexbench python=3.11
conda activate lexbench
cd lighteval
pip install -e .[dev]
cd ..
pip install -r requirements.txt

# Set API keys for inference and evaluation.
# OpenAI key is mandatory for our expert-verified grader, which is based on GPT-4o
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

# Evaluating GPT-4o-mini on LExBench Open Question subset.
python -m lighteval endpoint litellm "${MODEL}" "community|lexbenchoq_open_question|0|0" --custom-tasks lighteval/community_tasks/lexbench_oq_evals.py --output-dir outputs_oq --save-details --use-chat-template

# Evaluating GPT-4o-mini on LExBench Multiple-Choice Question subset.
python -m lighteval endpoint litellm "${MODEL}" "community|lexbenchmcq_mcq_4_choices|0|0" --custom-tasks lighteval/community_tasks/lexbench_mcq_evals.py --output-dir outputs_mcq --save-details --use-chat-template
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

# Evaluating GPT-4o-mini on LExBench Open Question subset.
python -m lighteval vllm "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16" "community|lexbenchoq_open_question|0|0" --custom-tasks lighteval/community_tasks/lexbench_oq_evals.py --output-dir outputs_oq --save-details --use-chat-template

# Evaluating GPT-4o-mini on LExBench Multiple-Choice Question subset.
python -m lighteval vllm "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16" "community|lexbenchmcq_mcq_4_choices|0|0" --custom-tasks lighteval/community_tasks/lexbench_mcq_evals.py --output-dir outputs_mcq --save-details --use-chat-template
```

### Evaluating Reasoning LLMs with LiteLLM directly.
Reasoning LLMs generate both a <think> scratch pad and the final answer after </think>. To only evaluate the answer, we do not use lighteval for reasoning LLMs.
```shell
MODEL="deepseek-reasoner"
python litellm_eval.py --input_file data/LFQA_test.xlsx --cache_name r1 --llm $MODEL --output_file lexbench_oq_${MODEL}.csv --batch_size 2
python litellm_eval.py --input_file data/MCQA_test.xlsx --cache_name r1 --llm $MODEL --output_file lexbench_mcq_${MODEL}.csv --batch_size 2
```
- `MODEL` can be set to any model included in `MODEL_DICT` of `litellm_eval.py`, e.g., `o1`, `o3-mini`, `qwq-32b`.
- `--output_file`: DeepSeek-R1's answer to open/MC questions will be at `lexbench_oq_deepseek-reasoner.csv` and `lexbench_mcq_deepseek-reasoner.csv`

Then evaluate the answers using our expert-verified LLM judge.
```shell
MODEL="deepseek-reasoner"
python customized_judge_async.py --input_file lexbench_oq_${MODEL}.csv --output_file lexbench_oq_${MODEL}_graded.csv --async_call --cache_name gpt4o --llm gpt-4o
```
- `--input_file`: Grade DeepSeek-R1's answer to open questions. Grading results at `lexbench_oq_deepseek-reasoner_graded.csv`

