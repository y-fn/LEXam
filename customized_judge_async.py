import os
import diskcache as dc
import time
import asyncio
import sys
import argparse
import re
import pandas as pd
from tqdm import tqdm
import openai
from litellm import completion, acompletion
import litellm
import traceback

# litellm.set_verbose = True
# litellm._turn_on_debug()

# openai_api = os.getenv("OPENAI_API_KEY")
# print(openai_api)

SKIP_MODELS = ['gold']

JUDGE_SYSTEM = "Act as a Judge specializing in the evaluation of Swiss law schools exams. Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning."
JUDGE_PROMPT = """Goal:
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
Judge the below case, give the brief reasoning process and the final grade.


Question:
```{question_fact}```

Reference Answer:
```{ref_answer}```

Model's Answer:
```[{model_answer}]```

Your Judgment:
"""


MODEL_DICT = {
    'o1': 'o1',
    'o3-mini': 'o3-mini',
    'gpt-4o-mini': 'openai/gpt-4o-mini',
    'gpt-4o': 'openai/gpt-4o',
    'gpt-4o-05': 'openai/gpt-4o',
    'deepseek-chat': 'deepseek/deepseek-chat',
    'deepseek-reasoner': 'deepseek/deepseek-reasoner',
    'together_v3': 'together_ai/deepseek-ai/DeepSeek-V3',
    'together_r1': 'together_ai/deepseek-ai/DeepSeek-R1',
    'sonnet-3.7-medium': "anthropic/claude-3-7-sonnet-20250219",
}
INPUT_COST_DICT = {
    'o1': 15,
    'o3-mini': 1.1,
    'gpt-4o-mini': 0.15,
    'gpt-4o': 2.5,
    'gpt-4o-05': 2.5,
    'deepseek-chat': 0.27,
    'deepseek-reasoner': 0.55,
    'together_v3': 1.25,
    'together_r1': 3,
    'sonnet-3.7-medium': 3,
}
OUTPUT_COST_DICT = {
    'o1': 60,
    'o3-mini': 4.4,
    'gpt-4o-mini': 0.6,
    'gpt-4o': 10,
    'gpt-4o-05': 10,
    'deepseek-chat': 1.10,
    'deepseek-reasoner': 2.19,
    'together_v3': 1.25,
    'together_r1': 7,
    'sonnet-3.7-medium': 15,
}
GENE_ARGS_DICT = {
    'deepseek-chat': {'temperature': 0, 'max_tokens': 4096},
    'deepseek-chat-05': {'temperature': 0.5, 'max_tokens': 4096, 'n': 2},
    'together_v3': {'temperature': 0, 'max_tokens': 4096},
    'together_v3-05': {'temperature': 0.5, 'max_tokens': 4096, 'n': 2},
    'gpt-4o': {'temperature': 0, 'max_tokens': 4096},
    'gpt-4o-05': {'temperature': 0.5, 'max_tokens': 4096, 'n': 2},
}


def get_input_price(model, input_len=None):
    input_cost = input_len / 1000000 * INPUT_COST_DICT[model]
    return input_cost


def get_output_price(model, output_len=None):
    output_cost = output_len / 1000000 * OUTPUT_COST_DICT[model]
    return output_cost


class LiteLLMChat:
    def __init__(
            self,
            model_name: str = None,
            cache_path: str = "litellm_cache",
            cache_name: str = "cache",
            generation_args: dict = None,
            system_prompt: str = JUDGE_SYSTEM,
    ):
        self.model_name = model_name
        self.cache_path = os.path.join(cache_path, f"{cache_name}.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.generation_args = generation_args
        self.system_prompt = system_prompt

    def ask(self, message: str):
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        with dc.Cache(self.cache_path, **cache_settings) as litellm_responses:
            if (self.model_name, message) in litellm_responses:
                reply_content = litellm_responses[(self.model_name, message)]
                print("Loaded from cache")
                input_price, output_price, input_token_num, output_token_num = 0, 0, 0, 0
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ]
                chat = self._send_request(messages)
                reply_content = {
                    'response': [choice.message.content for choice in chat.choices],
                    'response_reasoning': [choice.message.reasoning_content if hasattr(choice.message, 'reasoning_content') else None for choice in chat.choices],
                }
                litellm_responses[(self.model_name, message)] = reply_content
                input_token_num = chat.usage.prompt_tokens
                input_price = get_input_price(self.model_name, input_token_num)
                output_token_num = chat.usage.completion_tokens
                output_price = get_output_price(self.model_name, output_token_num)

        return reply_content, input_price, input_token_num, output_price, output_token_num

    def _send_request(self, messages):
        sleep_time_values = (5, 10, 30, 60, 120)
        arg_dict = {
            'model': MODEL_DICT[self.model_name],
            'messages': messages,
            **self.generation_args,
        }
        for i in range(len(sleep_time_values)):
            try:
                return completion(**arg_dict)
            except Exception as e:
                sleep_time = sleep_time_values[i]
                print(
                    f"Request to LiteLLM failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)

        return completion(**arg_dict)


def process_judge_response_grade_gpt(x):
    search = re.search(r"\[\[(\d.\d)\]\]", x)
    answer = float(search.group(1) if search else 0)
    if answer > 1 or answer < 0:
        answer = 0
    return answer


async def achat(model, messages, generation_args):
    output = await acompletion(model=MODEL_DICT[model], messages=messages, **generation_args)
    responses = [choice.message.content for choice in output.choices]
    input_token_num = output.usage.prompt_tokens
    output_token_num = output.usage.completion_tokens
    try:
        reasoning_content = [choice.message.reasoning_content for choice in output.choices]
    except Exception as e:
        reasoning_content = None
    return responses, reasoning_content, input_token_num, output_token_num


def batchify(lst, batch_size):
    """Split the list `lst` into sublists of size `batch_size`."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


async def create_answers_async(model, messages, cache_path, generation_args, batch_size=2):
    # async answering
    batched_msgs = batchify(messages, batch_size)
    total_input_tok_num = 0
    total_output_tok_num = 0
    print("{} batches to run.".format(len(batched_msgs)))
    all_answers = []
    cache_settings = dc.DEFAULT_SETTINGS.copy()
    cache_settings["eviction_policy"] = "none"
    cache_settings["size_limit"] = int(1e12)
    cache_settings["cull_limit"] = 0
    error_batches = []
    with dc.Cache(cache_path, **cache_settings) as litellm_responses:
        for i, batch in tqdm(enumerate(batched_msgs), total=len(batched_msgs)):
            mapping_list = []
            cache_miss_msgs = []
            cache_hit_responses = []
            for msg_in_batch in batch:
                if (model, msg_in_batch) in litellm_responses:
                    mapping_list.append(len(cache_hit_responses) + 1)
                    cache_hit_responses.append(litellm_responses[(model, msg_in_batch)]['response'])
                else:
                    mapping_list.append(- len(cache_miss_msgs) - 1)
                    cache_miss_msgs.append(msg_in_batch)

            if len(cache_miss_msgs) == 0:
                all_answers.extend(cache_hit_responses)
                print(f"Batch {i} entirely Loaded")
            else:
                try:
                    api_responses = await asyncio.gather(*[achat(model, m, generation_args) for m in cache_miss_msgs])
                    answers, reasoning_contents, input_tok_nums, output_tok_nums = zip(*api_responses)
                    total_input_tok_num += sum(input_tok_nums)
                    total_output_tok_num += sum(output_tok_nums)
                    for msg, res, reasoning in zip(cache_miss_msgs, answers, reasoning_contents):
                        litellm_responses[(model, msg)] = {'response': res, 'response_reasoning': reasoning}
                    merged_responses = []
                    for idx in mapping_list:
                        if idx > 0:
                            merged_responses.append(cache_hit_responses[idx - 1])
                        else:
                            merged_responses.append(answers[- idx - 1])
                    all_answers.extend(merged_responses)
                    print(f"Batch {i} Done")
                except Exception as e:
                    print(f"Batch {i} Error while gathering answers: {e}")
                    error_batches.append(i)
                    traceback.print_exc()

    input_price = get_input_price(model, total_input_tok_num)
    output_price = get_output_price(model, total_output_tok_num)
    return all_answers, error_batches, input_price + output_price


def main(args):
    df = pd.read_csv(args.input_file)
    if args.sample > 0:
        df = df.iloc[:1, :]
    model_names = []
    for column in df.columns:
        if '_answer' in column:
            model_name = column.replace('_answer', '')
            if model_name not in SKIP_MODELS:
                model_names.append(model_name)

    fact_questions = df['Facts_Question'].tolist()
    gold_answers = df['gold_answer'].tolist()
    litellm_chat = LiteLLMChat(model_name=args.llm, cache_name=args.cache_name, generation_args=GENE_ARGS_DICT[args.llm])
    for model in tqdm(model_names, total=len(model_names), desc="Model Progress"):
        model_answers = df[f'{model}_answer'].tolist()
        if '</think>' in model_answers[0]:
            model_answers = [a.split('</think>')[1].strip('\n ') for a in model_answers]
        prompts = []
        assert len(fact_questions) == len(gold_answers) == len(model_answers)
        for q, ref, answer in zip(fact_questions, gold_answers, model_answers):
            prompts.append(JUDGE_PROMPT.format(question_fact=q, ref_answer=ref, model_answer=answer))

        responses = []
        scores = []
        for p in tqdm(prompts, total=len(prompts), desc=f"Judging {model}"):
            reply_content, _, _, _, _ = litellm_chat.ask(p)
            response_text = reply_content['response']
            scores.append([process_judge_response_grade_gpt(t) * 100 for t in response_text])
            responses.append(response_text)

        df[f'{model}_{args.llm}_judge'] = responses
        df[f'{model}_{args.llm}_scores'] = scores
        df.to_csv(args.output_file, encoding='utf-8', index=False)


async def async_main(args):
    df = pd.read_csv(args.input_file)
    if args.sample > 0:
        df = df.iloc[:args.sample, :]
    model_names = []
    for column in df.columns:
        if '_answer' in column:
            model_name = column.replace('_answer', '')
            if model_name not in SKIP_MODELS:
                model_names.append(model_name)

    fact_questions = df['Facts_Question'].tolist()
    gold_answers = df['gold_answer'].tolist()
    for model in tqdm(model_names, total=len(model_names), desc="Model Progress"):
        model_answers = df[f'{model}_answer'].tolist()
        if '</think>' in model_answers[0]:
            model_answers = [a.split('</think>')[1].strip('\n ') for a in model_answers]
        prompts = []
        assert len(fact_questions) == len(gold_answers) == len(model_answers)
        for q, ref, answer in zip(fact_questions, gold_answers, model_answers):
            prompts.append([
                {'role': 'system', 'content': JUDGE_SYSTEM},
                {'role': 'user', 'content': JUDGE_PROMPT.format(question_fact=q, ref_answer=ref, model_answer=answer)},
            ])

        total_cost = 0
        while True:
            reply_texts, error_batches, cost = await create_answers_async(args.llm, prompts, cache_path=f"litellm_cache/{args.cache_name}.diskcache", generation_args=GENE_ARGS_DICT[args.llm], batch_size=args.batch_size)
            total_cost += cost
            if len(error_batches) == 0:
                break
        print(f"Total cost {total_cost}")
        responses = reply_texts
        scores = [[process_judge_response_grade_gpt(t) * 100 for t in texts] for texts in reply_texts]

        df[f'{model}_{args.llm}_judge'] = responses
        df[f'{model}_{args.llm}_scores'] = scores
        df.to_csv(args.output_file, encoding='utf-8', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--cache_name", type=str, default='deepseek')
    parser.add_argument("--llm", type=str, default="deepseek-chat")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--async_call", action='store_true', default=False)
    args = parser.parse_args()

    if args.async_call:
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(async_main(args))
    else:
        main(args)







