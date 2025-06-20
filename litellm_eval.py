import os
import diskcache as dc
import time
import asyncio
import sys
import argparse
import pandas as pd
import ast
from tqdm import tqdm
from litellm import completion, acompletion


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


MODEL_DICT = {
    'o1': 'o1',
    'o3-mini': 'o3-mini',
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4o': 'gpt-4o',
    'deepseek-chat': 'together_ai/deepseek-ai/DeepSeek-V3',
    'deepseek-reasoner': "together_ai/deepseek-ai/DeepSeek-R1",
    'qwq-32b': 'together_ai/Qwen/QwQ-32B',
    'sonnet-3.7-high': "anthropic/claude-3-7-sonnet-20250219",
    'gemini-2.5-pro': "gemini/gemini-2.5-pro-preview-03-25",
    'llama_405': "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    'llama_70': "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    'llama4_maverick': "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    'gemma3': "together_ai/google/gemma-3-12b-it",
    'gpt-4.1-mini': 'gpt-4.1-mini-2025-04-14',
    'gpt-4.1-nano': 'gpt-4.1-nano-2025-04-14',
    'qwen3_235B': 'together_ai/Qwen/Qwen3-235B-A22B-fp8-tput'
}
INPUT_COST_DICT = {
    'o1': 15,
    'o3-mini': 1.1,
    'gpt-4o-mini': 0.15,
    'gpt-4o': 2.5,
    'gpt-4.1-mini': 0.4,
    'gpt-4.1-nano': 0.1,
    # 'deepseek-chat': 0.27,
    # 'deepseek-reasoner': 0.55,
    'deepseek-chat': 1.25,
    'deepseek-reasoner': 3,
    'qwq-32b': 1.2,
    'sonnet-3.7-high': 3,
    'gemini-2.5-pro': 1.25,
    'llama_405': 3.5,
    'llama_70': 0.88,
    'llama4_maverick': 0.27,
    'gemma3': 0.3,
    'qwen3_235B': 0.2,
}
OUTPUT_COST_DICT = {
    'o1': 60,
    'o3-mini': 4.4,
    'gpt-4o-mini': 0.6,
    'gpt-4o': 10,
    'gpt-4.1-mini': 1.6,
    'gpt-4.1-nano': 0.4,
    # 'deepseek-chat': 1.10,
    # 'deepseek-reasoner': 2.19,
    'deepseek-chat': 1.25,
    'deepseek-reasoner': 7,
    'qwq-32b': 1.2,
    'sonnet-3.7-high': 15,
    'gemini-2.5-pro': 2.5,
    'llama_405': 3.5,
    'llama_70': 0.88,
    'llama4_maverick': 0.85,
    'gemma3': 0.3,
    'qwen3_235B': 0.6,
}
GENE_ARGS_DICT = {
    'gpt-4.1-mini': {'temperature': 0, 'max_tokens': 4096},
    'gpt-4.1-nano': {'temperature': 0, 'max_tokens': 4096},
    'gpt-4o-mini': {'temperature': 0, 'max_tokens': 4096},
    'deepseek-reasoner': {'temperature': 0.6, 'max_tokens': 8192},
    'qwq-32b': {'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 8192},
    'o3-mini': {'reasoning_effort': 'high', 'max_tokens': 8192},
    'sonnet-3.7-high': {'reasoning_effort': 'high', 'max_tokens': 8192},
    'gemini-2.5-pro': {'max_tokens': 8192},
    'deepseek-chat': {'temperature': 0, 'max_tokens': 4096},
    'llama_405': {'temperature': 0, 'max_tokens': 4096},
    'llama_70': {'temperature': 0, 'max_tokens': 4096},
    'llama4_maverick': {'temperature': 0, 'max_tokens': 4096},
    'gemma3': {'temperature': 0, 'max_tokens': 4096},
    'qwen3_235B': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'max_tokens': 8192},
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
    ):
        self.model_name = model_name
        self.cache_path = os.path.join(cache_path, f"{cache_name}.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.generation_args = generation_args

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
                messages = [{"role": "user", "content": message}]
                chat = self._send_request(messages)
                reply_content = {
                    'response': chat.choices[0].message.content,
                    'response_reasoning': chat.choices[0].message.reasoning_content,
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
            'model': self.model_name,
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


async def achat(model, messages, generation_args):
    output = await acompletion(model=MODEL_DICT[model], messages=messages, **generation_args)
    input_token_num = output.usage.prompt_tokens
    output_token_num = output.usage.completion_tokens
    try:
        reasoning_content = output.choices[0].message.reasoning_content
    except Exception as e:
        reasoning_content = None
    return output.choices[0].message.content, reasoning_content, input_token_num, output_token_num


def batchify(lst, batch_size):
    """Split the list `lst` into sublists of size `batch_size`."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


async def create_answers_async(model, messages, cache_path, generation_args, batch_size=5):
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

    input_price = get_input_price(model, total_input_tok_num)
    output_price = get_output_price(model, total_output_tok_num)
    return all_answers, error_batches, input_price + output_price


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=['mcq_numbers', 'mcq_letters', 'qa'])
    parser.add_argument("--cache_name", type=str, default='openai')
    parser.add_argument("--question_field", type=str, default='question')
    parser.add_argument("--answer_field", type=str, default='answer')
    parser.add_argument("--llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    input_df = pd.read_excel(args.input_file)
    prompts = []
    questions = input_df[args.question_field].tolist()
    course_names = input_df['course'].tolist()
    if args.task_type == 'mcq_letters':
        letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        choices = [ast.literal_eval(c) if isinstance(c, str) else c for c in input_df[args.choices_field].tolist()]
        formatted_questions = []
        for q, c, course_name in zip(questions, choices, course_names):
            formated_question = q
            for i, c in enumerate(c):
                formated_question += f'\n{letter_list[i]}. {c}'
            formatted_questions.append(formated_question)
        for q, course_name in zip(formatted_questions, course_names):
            prompts.append(MCQ_PROMPT['letters'].format(course_name=course_name, question=q))
    elif args.task_type == 'mcq_numbers':
        choices = [ast.literal_eval(c) if isinstance(c, str) else c for c in input_df[args.choices_field].tolist()]
        formatted_questions = []
        for q, c, course_name in zip(questions, choices, course_names):
            formated_question = q
            for i, c in enumerate(c):
                formated_question += f'\n{i + 1}. {c}'
            formatted_questions.append(formated_question)
        for q, course_name in zip(formatted_questions, course_names):
            prompts.append(MCQ_PROMPT['numbers'].format(course_name=course_name, question=q))
    else:
        for q, course_name in zip(questions, course_names):
            prompts.append(QA_PROMPT.format(course_name=course_name, question=q))

    gold_answers = input_df[args.answer_field].tolist()
    if args.sample > 0:
        prompts = prompts[:args.sample]
        gold_answers = gold_answers[:args.sample]

    total_cost = 0
    responses, err_batches, cost = await create_answers_async(
        args.llm,
        prompts,
        cache_path=os.path.join('litellm_cache', f"{args.cache_name}.diskcache"),
        generation_args=GENE_ARGS_DICT[args.llm],
        batch_size=args.batch_size,
    )
    total_cost += cost
    print("Error Batches", err_batches)
    print(f"Total cost {total_cost}")

    print(len(prompts), len(responses))
    output_df = pd.DataFrame({
        'prompt': [p[0]['content'] for p in prompts],
        'gold_answer': gold_answers,
        f'{args.llm}_answer': responses,
    })

    output_df.to_csv(args.output_file, encoding='utf-8', index=False)


if __name__ == '__main__':
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

