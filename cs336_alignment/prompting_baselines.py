import json
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)

PROMPT_TYPE = ["question_only", "r1_zero", "r1_zero_three_shot_gsm8k"]

FILE_PATH = "../data/gsm8k/test.jsonl"


NUM_EXAMPLES = None

def load_gsm8k(file_path):
    examples = []

    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)

        return examples

def find_ground_truth(answer: str) -> str:
    return answer.split('####')[-1].strip()

def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, 'r', encoding = 'utf-8') as f:
        return f.read()

def build_prompt(template: str, question: str) -> str:
    return template.format(question = question)

def prepare_batch(
    examples: list[dict],
    template: str,
    num_examples: int | None = None,
):
    if num_examples is not None:
        examples = examples[ : num_examples]

    prompts = []
    ground_truths = []

    for example in examples:
        question = example['question']

        prompt = build_prompt(
            template,
            question,
        )

        ground_truth = find_ground_truth(
            example['answer']
        )

        prompts.append(prompt)
        ground_truths.append(ground_truth)

    return prompts, ground_truths

def get_sampling_params(prompt_type: str) -> dict:
    sampling_params = {
        'temperature' : 1.0,
        'top_p' : 1.0,
        'max_tokens' : 512,
        'n' : 1,
        'seed' : 0,
    }

    if prompt_type in {'r1_zero', 'r1_zero_three_shot_gsm8k'}:
        sampling_params['stop'] = ['</answer>']
        sampling_params["include_stop_str_in_output"] = True
    
    return sampling_params

def get_reward_fn(prompt_type: str):
    if prompt_type == "question_only":
        return question_only_reward_fn
    
    return r1_zero_reward_fn

def main():

    examples = load_gsm8k(FILE_PATH)

    server = VLLMServer(
        model_id = 'allenai/OLMo-2-0425-1B',
        gpu = 0,
        seed = 0,
        gpu_memory_utilization = 0.8,
        startup_timeout=1200,
    )
    server.start()

    for prompt_type in PROMPT_TYPE:

        prompt_path = 'prompts/' + prompt_type + '.prompt'

        prompts, ground_truths = prepare_batch(
            examples,
            load_prompt_template(prompt_path),
            NUM_EXAMPLES,
        )

        completions = server.generate_completions(
            prompts = prompts,
            sampling_params = get_sampling_params(prompt_type),
        )

        reward_fn = get_reward_fn(prompt_type)

        reward_count = 0

        for i in range(len(prompts)):
            response = completions[i].text
            ground_truth = ground_truths[i]

            result = reward_fn(response, ground_truth)

            if result["reward"] == 1.0:
                reward_count += 1

            print("response:")
            print(response)

            print("ground truth:", ground_truth)
            print("result:", result)
            print("=" * 80)

        reward_rate = reward_count / len(prompts) * 100

        print(f"\n{prompt_type} result:")
        print(f"reward=1: {reward_count}/{len(prompts)} ({reward_rate:.2f}%)")
        print("#" * 80)


if __name__ == '__main__':
    main()