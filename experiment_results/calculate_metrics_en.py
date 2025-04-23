import json
import time
import random
import openai
import argparse
from tqdm import tqdm
from openai import OpenAI

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--vote_times', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=10)
    parser.add_argument('--response_file_name', type=str, required=True)
    parser.add_argument('--result_save_path', type=str, default='results.json')
    parser.add_argument('--metric_save_path', type=str, default='non_hallucination_rate.txt')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--reference_file_name', type=str, required=True)
    args = parser.parse_args()
    return args

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 50,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""
 
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
 
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
 
            # Retry on specific errors
            except Exception as e:
                # Increment retries
                num_retries += 1
                # print(e)
 
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
 
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
 
                # Sleep for the delay
                time.sleep(delay)
                # time.sleep(30)
 
            # Raise exceptions for any errors not specified
            # except Exception as e:
            #     raise e
 
    return wrapper
    
@retry_with_exponential_backoff
def chat_completion_with_backoff(
    model,
    messages,
    temperature,
    top_p,
    n,
    max_tokens,
    api_key,
):
    try:
        client = OpenAI(api_key=api_key, timeout=30)
        return client.chat.completions.create(model=model, messages=messages, temperature=temperature, top_p=top_p, n=n, max_tokens=max_tokens)
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        raise e

def get_prompt(sample, resource):
    ref = resource[sample['question_id']]
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': ''}
    ]
    messages[-1]['content'] = '''I need you to evaluate whether a language model's response contains hallucination. I will give you some example correct answers. Please follow these criteria:
1. First, determine whether the model's response is fluent. If the output is not fluent natural language (e.g., contains gibberish), it should be judged as hallucination.
2. Then, assess whether the model directly answers the question. If it contains many correct facts but dodges the question, it should be judged as hallucination.
3. If the model's response cannot be inferred from or is inconsistent with the reference answers, it should be considered a hallucination.
4. If the model's response is supported by any one of the reference answers, then it should not be considered a hallucination.
5. If the model's response is not directly supported by a reference answer, you should infer whether it has similar meaning. If it does, then it should not be considered a hallucination.
6. If one of the reference answers includes a statement like “This question cannot be answered,” then a response like “I don't know” is acceptable and should not be considered a hallucination.'''
    
    messages.append({'role': 'assistant', 'content': "Understood. I will judge whether the model's response contains hallucination based on the reference answers and the criteria you provided. Please provide the question, correct answers, and the model's response."})
    messages.append({'role': 'user', 'content': ''})

    # assert sample['question'] == ref['Question'], print(sample['question'], ref['Question'])
    assert sample['question_id'] == ref['question_id']

    user_input_for_judging = 'Question:{}\n\n'.format(ref['Question'].strip())
    user_input_for_judging += 'Reference correct answers:\n'
    if 'Best Answer1' in ref:
        count = 1
        for i in range(1,5):
            if 'Best Answer{}'.format(str(i)) in ref:
                correct_answer_key = 'Best Answer{}'.format(str(i))
                if ref[correct_answer_key] != '':
                    user_input_for_judging += '{}. {}\n'.format(str(count), ref[correct_answer_key].strip())
                    sample['Best_Answer{}'.format(str(i))] = ref[correct_answer_key].strip()
                    count += 1
    else:
        user_input_for_judging += '1. {}\n'.format(ref['Best Answer'].strip())
        sample['Best_Answer'] = ref['Best Answer'].strip()

    user_input_for_judging += '\nModel response:\n'
    user_input_for_judging += '{}\n\n'.format(sample['response'].strip())
    user_input_for_judging += 'Now please judge whether the response contains hallucination. Just answer "yes" or "no"'

    if ref.get("language_sensitive", False):
        user_input_for_judging = (
            "Note: The following question involves Chinese poetry, idioms, or culturally specific language. "
            "The model’s response may contain Chinese terms or be multilingual. This is acceptable.\n\n"
        ) + user_input_for_judging
    
    messages[-1]['content'] = user_input_for_judging

    return sample, messages

def calculate(args, resource):
    with open(args.response_file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scored_outputs = []
    correct_count = 0
    for i, item in enumerate(tqdm(data)):
        print(f"Processing item {i+1}/{len(data)}")
        if 'question_id' not in item or item['question_id'] not in resource:
            print(f"Warning: Item {i} missing question_id or not found in reference")
            continue
        sample, messages = get_prompt(item, resource)
        max_try = 5
        try_count = 0
        invalid_judge = False
        while True:
            try_count += 1
            print("Start to judge...")
           
            responses = chat_completion_with_backoff(
                model="gpt-4o-mini",
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                n=args.vote_times,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
            )
            # check output
            flag = True
            
        
            for choice in responses.choices:
                response_content = choice.message.content.strip().lower()
                if response_content != 'yes' and response_content != 'no':
                    flag = False
                    break
            if flag:
                break
            if try_count >= max_try:
                print(f"Item {i+1}/{len(data)}: Try count received...{try_count}")
                invalid_judge = True
                break
            time.sleep(1)
        time.sleep(2)

        if invalid_judge is False:
            outputs = []
            for choice in responses.choices:
                outputs.append(choice.message.content)
            
            if outputs.count('yes') > 2:
                sample['is_hallucination'] = True
            else:
                sample['is_hallucination'] = False
                if sample['response'] != '':
                    correct_count += 1
                else:
                    sample['is_hallucination'] = True
            scored_outputs.append(sample)
        else:
            sample['is_hallucination'] = "Invalid_Judge"
            scored_outputs.append(sample)
            print("loop finished")

    assert len(data) == len(scored_outputs)

    try:
        with open(args.result_save_path, 'w', encoding='utf-8') as f:
            json.dump(scored_outputs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        # Backup save attempt
        with open(f"{args.result_save_path}.backup", 'w', encoding='utf-8') as f:
            json.dump(scored_outputs, f, indent=2, ensure_ascii=False)
            
    with open(args.metric_save_path, 'w', encoding='utf-8') as f:
        f.write('Non hallucination rate: {:.2f}%'.format(correct_count/len(data)*100))

if __name__ == '__main__':
    args = get_args()
    # Load reference data
    with open(args.reference_file_name, 'r', encoding='utf-8') as f:
        resource = {item['question_id']: item for item in json.loads(f.read())}

    print('Evaluating hallucination for {}...'.format(args.response_file_name))
    calculate(args, resource)