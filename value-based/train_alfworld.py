import random
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup, AdamW

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import LoraConfig, get_peft_model, PeftModel
from omegaconf import OmegaConf, DictConfig
import llfbench as gym
import concurrent.futures

from torch.utils.data import DataLoader
from transformers import default_data_collator,LlamaForCausalLM, LlamaTokenizer
from AIF.env.frozen_lake import FrozenLakeEnv
from torch.utils.data import RandomSampler

import json
from datetime import datetime
import os
from pathlib import Path
import time
print('1111111')
hislen=3
ROOT_PATH = str(Path(__file__).resolve().parent)
# hard_tasks=['put some cellphone on sidetable.', 'put a cellphone in bed.', 'put some cd on dresser.', 'put some cellphone on sidetable.', 'put a cellphone in sidetable.', 'put some mug on sidetable.', 'put a fork in drawer.', 'put some fork on drawer.', 'put a cellphone in desk.', 'put a fork in drawer.', 'put a cellphone in desk.', 'put some fork on drawer.', 'put some mug on sidetable.', 'put some cellphone on sidetable.', 'put a cellphone in sidetable.', 'put some cellphone on sidetable.', 'put a cellphone in bed.', 'put some cd on dresser.']
hard_tasks=['put some cellphone on sidetable.', 'put a cellphone in bed.', 'put some fork on drawer.', 'put a cellphone in sidetable.', 'put some cd on dresser.']
test_envs=['Task: put a soapbottle in toilet.', 'Task: put a handtowel in garbagecan.', 'Task: put some cellphone on sidetable.', 'Task: put some candle on toilet.', 'Task: put a cellphone in sidetable.', 'Task: put some soapbottle on toilet.', 'Task: put a candle in toilet.']
test_envs=['put some cellphone on armchair.', 'put some candle on sidetable.', 'put some candle on toilet.', 'put a cellphone in armchair.', 'put some soapbottle on toilet.', 'put some cellphone on bed.', 'put some cellphone on sidetable.', 'put a cloth in toilet.', 'put some cellphone on desk.', 'put a cellphone in drawer.', 'put some spraybottle on countertop.', 'put some pillow on sofa.', 'put some soapbottle on cart.', 'put some spraybottle on toilet.', 'put some laptop on armchair.', 'put a candle in sidetable.', 'put a cellphone in bed.', 'put a handtowel in garbagecan.', 'put some cd on dresser.', 'put some box on armchair.', 'put a cd in dresser.', 'put a spraybottle in countertop.', 'put a candle in toilet.', 'put a soapbottle in toilet.', 'put a laptop in armchair.', 'put a cellphone in sidetable.', 'put some mug on sidetable.', 'put a fork in drawer.', 'put a pillow in sofa.', 'put some fork on drawer.', 'put a cellphone in desk.']
test_envs = [task for task in test_envs if task not in hard_tasks]
good_example_t=100
bad_example_t=0

env_ind=0
test_env_ind=0
import numpy as np
trajectories=[]
def count_substring_case_insensitive(string, substring):
    return string.lower().count(substring.lower())

# 示例
import re

def extract_before_phrase_regex(text, phrase):
    # 使用正则表达式查找短语，并获取其位置
    match = re.search(re.escape(phrase), text)
    # 如果找到了匹配，返回匹配之前的内容
    if match:
        return text[:match.start()]
    # 如果没有找到匹配，返回整个字符串
    return text


def extract_task(description):
    # 使用正则表达式匹配 "Your task is to:" 开头，直到句号"."结束的部分
    match = re.search(r"Your task is to: (.*?\.)", description)
    # 如果找到匹配的字符串，则返回匹配的部分，否则返回 None
    if match:
        return 'Task: '+match.group(1)  # 返回第一个捕获组的内容
    else:
        return None

# 示例
# print(result)  # 输出 "Here is some introductory text. "

def process_text2action(s,choices):
    maxx=-1
    a='look'
    for choice in choices:
        num_a= count_substring_case_insensitive(s,choice)
        if num_a>=maxx:
            maxx=num_a
            a=choice
    if maxx==-1:
        return random.sample(choices,1)[0]
    else:
        return a

    # num_left = len(s.lower().split('left')) - 1
    # num_right = len(s.lower().split('right')) - 1
    # num_up = len(s.lower().split('up')) - 1
    # num_down = len(s.lower().split('down')) - 1
    # action_set = [num_left, num_down, num_right, num_up]
    # if sum(action_set) == 0:
    #     return random.choice([0, 1, 2, 3])
    # else:
    #     return np.argmax(action_set)


class LinearProb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d,64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        # x=x.to(torch.float32)
        o1 = self.linear1(x)
        o2 = self.linear2(F.relu(o1))
        o3 = self.linear3(F.relu(o2))
        # print(o3.type)
        return o3

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

from transformers import StoppingCriteria, StoppingCriteriaList
class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

def get_stopping_criteria(stop_words, tokenizer, stop_on_newline=False):
    stopping_criteria = []
    if stop_words and tokenizer.eos_token:
        stop_words.append(tokenizer.eos_token)
    if stop_words and stop_on_newline:
        stop_words.append("\n")
    print("stop_words:", stop_words)
    if stop_words:
        stopping_criteria.append(
            EndOfFunctionCriteria(0, stop_words, tokenizer)
        )
    return StoppingCriteriaList(stopping_criteria)

def stop_at_stop_token(decoded_string, stop_tokens):
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index

    return decoded_string[:min_stop_index]

class LLM_Generate_Instance:
    def __init__(self, config, tkn, policy,  num_gens, gen_bz, device,stop_words=["\n"],max_new_tokens=30):
        self.config = config
        self.tokenizer = tkn
        self.model = policy

        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.stop_words = stop_words#["\n"]
        self.stopping_criteria = get_stopping_criteria(
            self.stop_words, self.tokenizer
        )
        self.num_gens = num_gens           # total number of generated response
        self.device = device
        self.temperature = 1.0#0.8#0.8#0.8
        self.batch_size = gen_bz           # splitted into how many batches with this batch size
        self.max_new_tokens = max_new_tokens
        self.top_p = 1.0#0.95#0.8#0.95
        self.return_hiddens = True
        self.debug = False
        self.layer=-1

    def get_actions(self, prompt: str, num_batches=None, temperature=None):
        assert self.num_gens % self.batch_size == 0
        if num_batches is None:
            num_batches = self.num_gens // self.batch_size
        if temperature is None:
            temperature=self.temperature

        top_k_strings, hiddens = [], []
        for _ in range(num_batches):
            s, h = self.get_actions_single_batch(prompt, temperature)
            top_k_strings.extend(s)
            if h is not None:
                hiddens.extend(h)
        return top_k_strings, hiddens

    def get_actions_single_batch(self, prompt: str, temperature: float):

        with torch.no_grad():
            encoded_output = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
            prompt_ids = encoded_output.input_ids[0].to(self.device)
            atten = encoded_output.attention_mask.to(self.device)

            # prompt_ids = (
            #     self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
            #         .input_ids[0].to(self.device)
            # )
            start_time = time.time()
            self.stopping_criteria[0].start_length=len(prompt_ids)+1

            model_output = self.model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=atten,
                num_return_sequences=self.batch_size,
                # EOS parameters
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.eos_token_id,
                # Sampling
                do_sample=True if self.temperature > 0 else False,
                temperature=temperature,
                top_p=self.top_p,
                # Output parameters
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=self.return_hiddens,  # gen_toks x layers x (k x input_len x D)
                # Cache
                use_cache=True,
            )

            # delicate code: due to the annoying unexchangability between encode() and string concatenation()
            max_gen_length = len(model_output.scores)
            pre_decoded_sequences = self.tokenizer.batch_decode(
                model_output.sequences
            )  # with prompt and suffixes
            sequences_wp = [
                _[len(prompt) :] for _ in pre_decoded_sequences
            ]  # without prompt, with suffixes
            top_k_strings = [
                stop_at_stop_token(_, self.stop_words)
                for _ in sequences_wp
            ]  # without prompt, without suffixes

            useful_lengths = [
                self.tokenizer.encode_plus(
                    prompt + _, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].size(1)
                for _ in top_k_strings
            ]  # token length with prompt, without suffixes
            useful_lengths = [
                min(_ - len(prompt_ids), max_gen_length) for _ in useful_lengths
            ]  # token length without prompt, without suffixes

            if self.return_hiddens:
                hiddens = []
                for i, l in enumerate(useful_lengths):  # index by action
                    hiddens.append(
                        model_output.hidden_states[l - 1][self.layer][i, 0, :]
                        .detach()
                        .cpu().float()
                        .numpy()
                    )
            else:
                hiddens = None

        if self.debug:
            print("generate top-k time: " + str(time.time() - start_time))

        return top_k_strings, hiddens


Action_prompt='Please select an action from the admissible actions. At each step of this sequence, you will be told what actions are allowed. You should only select an action that is allowed. \n Action:'
phrase = "You are allowed to take the following actions:"
Action_example='''A case of step-by-step action:

Your task is to: put a candle in toilet..\n
Step: 0, Current Observation: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1. Action: look. \n
Step: 1, Current Observation: You are in the middle of a room. Looking quickly around you, you see nothing.. , Action: go to toilet 1. \n
Step: 2, Current Observation: You arrive at loc 7. On the toilet 1, you see a cloth 1, and a toiletpaper 2.. , Action: go to countertop 1. \n
Step: 3, Current Observation: You arrive at loc 16. On the countertop 1, you see a candle 1.. , Action: take candle 1 from countertop 1. \n
Step: 4, Current Observation: You pick up the candle 1 from the countertop 1.. , Action: go to toilet 1. \n
Step: 5, Current Observation: You arrive at loc 7. On the toilet 1, you see a cloth 1, and a toiletpaper 2.. , Action: put candle 1 in/on toilet 1. \n
'''

def feat_generate(state_text, tkn, p, gen_action, device,task):
        phrase = "You are allowed to take the following actions:"

    # action_choices = ['left', 'down', 'right', 'up']
        feat_list = []
        state_text = extract_before_phrase_regex(state_text, phrase)
        for i, a in enumerate(gen_action):
            input_action = a
            state_info = f"{task},  Action: {input_action}, for Observation: {state_text},"
            # print(state_info)
            feat_list.append(state_info)
        _feat_encoded = tkn(feat_list, return_tensors='pt', padding=True, max_length=40)
        feat_encoded = {k: v.to(device) for k, v in _feat_encoded.items()}
        feat_output = p(**feat_encoded, output_hidden_states=True)
        feat = feat_output.hidden_states[-1][:,0,:]#.mean(-2)
        return feat
def adjust_list_length(lst, desired_length=30, fill_value="nope"):
    """
    Adjust the length of a list to the desired length.
    If the list is longer than or equal to the desired length, it will be truncated.
    If the list is shorter, it will be extended by copying the list until the desired length is reached.

    Parameters:
        lst (list): The list to adjust.
        desired_length (int): The desired length of the list.
        fill_value (any): The value to use for filling if the list is empty.

    Returns:
        list: The adjusted list with the exact desired length.
    """
    # 如果列表长度大于等于期望长度，裁剪列表
    if len(lst) >= desired_length:
        return lst[:desired_length]
    # 如果列表为空，用fill_value填充
    else:
        # 计算需要填充的次数
        times_to_copy = (desired_length // len(lst)) + 1
        extended_list = lst * times_to_copy
        return extended_list[:desired_length]

def get_thought_actions(llm_policy,llm_thought_policy,instruction,state_text,Action_prompt,history=''):
    # actions_, _ = llm_policy.get_actions(instruction+'\n'+'Current observation:'+state_text+'\n'+Action_prompt)
    # print(actions_)
    match = re.search(r'Your task is to: [^.]+\.?', instruction)

    if match:
        #print(match.group())
        task='Task: '+match.group()
    else:
        raise Exception("No match found for a sentence starting with 'Your task is to:'")
    actions_, _ = llm_policy.get_actions(task+'\n'+'Current observation:'+state_text+'\n'+Action_prompt)
    # print(actions_)
    print(task+'\n'+'Current observation:'+state_text+'\n'+Action_prompt)
    print(actions_)

    '''
    def get_actions_for_thought(instruction, state_text, thought, action_prompt,history, llm_policy):
        outputs = llm_policy.get_actions(
            instruction + '\n' + 'Current observation:' + state_text +
            '\n' + 'Thought hints: ' + thought + '\n' + action_prompt)
        return outputs
    # print(instruction)
    thought_examples = """
    A number of thought examples:
    Task: Put a spray bottle on the toilet, \n **The thought is:** First, find the spray bottle. It's likely to be in the cabinets, countertop, or sinkbasin. Check these areas one by one.\n
    Task: Find an apple and place it on the sidetable, \n **The thought is:** First, locate the apple. It's likely to be found in the fridge, dining tables, countertops, or garbage can. Search one location at a time.\n
    Task: Put a clean lettuce on the dining table, \n **The thought is:** First, find the lettuce. It is likely in the fridge, sinkbasin, or cabinets. Then, clean it before placing it on the table.\n
    Task: Your task is to: put a cellphone in sidetable. Looking quickly around you, you see a bed 2, a bed 1, a diningtable 1, a drawer 2, a drawer 1, a garbagecan 1, a shelf 14, a shelf 13, a shelf 12, a shelf 11, a shelf 10, a shelf 9, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, and a sidetable 1.\n **The thought is:** First, find the cellphone. It might be located in places like the drawer, bed, or shelves. Check these areas one by one. Once you have found the cellphone, go to the sidetable and place it there.\n\n
    Task: Your task is to: put a cellphone in sidetable. On the shelf 12, you see nothing. On the shelf 7, you see a alarmclock 1, and a mug 1.. On the shelf 2, you see nothing.. \n **The thought is:** You need to find the cellphone before you can place it in the sidetable. Continue searching the room by checking other locations such as the drawers, beds, and other shelves.\n\n
    Task: Your task is to: put a handtowel in garbagecan. You are in the middle of a room. Looking quickly around you, you see a countertop 1, a drawer 2, a drawer 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1. \n  **The thought is:** First, locate the handtowel. It's likely to be found on the handtowelholder 1, handtowelholder 2, or the towelholder. Check these locations one by one. Once you find the handtowel, take it and go to the garbagecan to place it there, completing the task.\n\n
    """

    """
    Case4: Task: Cool a pan and place it on the stoveburner, Thought: Find the pan first, likely on the stoveburner or in the cabinets. After locating it, cool it in the fridge before placing it on the stoveburner.\n
    Case5: Task: Place two credit cards in a dresser, Thought: First, find the first credit card. It could be in drawers or on the countertop. After placing it in the dresser, find the second credit card in the same location.\n
    Case6: Task: Examine a pen with the desklamp, Thought: Find the pen in drawers or on shelves, then locate the desklamp to use for examination.\n
    """

    # thoughts_example = "A thought case:\n  Your task is to put some spraybottle on toilet. You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1. \n>Thought: To solve the task, I need to find and take a sparybottle, then put it on toilet. First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.\n<END>"
    #thought_prompt=thoughts_example+'\n'+\

    thought_prompt=thought_examples+'Please directly give a thought suggestion about how to proceed with current observation for completing the task.\nOutput a concise thought in one sentence in the form of: **The thought is:** \n'
    outputs = llm_thought_policy.get_actions(task+' '+'History:' +history+'\n'+'Current observation:'+state_text+'\n'+thought_prompt)
    thoughts, hiddens = outputs
    # print(thoughts)
    # actions_=[]
    # for i, a in enumerate(thoughts):
    #     outputs = llm_policy.get_actions(
    #         instruction + '\n' + 'Current observation:' + state_text+  '\n'+'Thought hints: '+a+'\n' + Action_prompt)
    #     actions, hiddens = outputs
    #     actions_.append(actions)
    # return actions_

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务
        futures = [executor.submit(get_actions_for_thought, instruction, state_text, a, Action_prompt,history, llm_policy) for a
                   in thoughts]

        # 收集结果
        actions_ = []
        for future in concurrent.futures.as_completed(futures):
            actions, _ = future.result()
            actions_+=actions
    # print(thoughts)
    # print(actions_)
    '''
    return actions_
example_dict={}
trajectories_bala=[]
def explore_rollout_q(env, embed_tkn, embed_policy, Q,
                      llm_policy,llm_thought_policy, buffer,
                      capacity, device, action_normalize,epsilon=0.0,alg='dqn'):
    global example_dict,good_example_t,bad_example_t,env_ind,trajectories,trajectories_bala
    trajectory=[]
    # action_choices = ['left', 'down', 'right', 'up']
    # phrase = "You are allowed to take the following actions:"
    historys=[]

    Action_prompt='Please select an action from the admissible actions. \n Action:'

    # ref_agent = reference_response()
    total_reward = 0

    env_ret, info = env.reset()
    # obs, prompt, state_text = env_ret['obs'], env_ret['text'], env_ret['state_text']

    instruction = env_ret['instruction']
    task=extract_task(instruction)
    target_task=test_envs[env_ind]
    while target_task not in task:
        env_ret, info = env.reset()
        # obs, prompt, state_text = env_ret['obs'], env_ret['text'], env_ret['state_text']

        instruction = env_ret['instruction']
        task = extract_task(instruction)
    print(task)
    env_ind+=1
    if env_ind==len(test_envs):
        env_ind=0
        random.shuffle(test_envs)

    admissible_commands = info['admissible_commands']

    # obs, prompt, state_text = env_ret['obs'], env_ret['text'], env_ret['state_text']
    done = False


    state_text = env_ret['observation']
    len_welcome = len('-= Welcome to TextWorld, ALFRED! =-\n\n')
    if '-= Welcome to TextWorld, ALFRED! =-\n\n' in state_text:
        state_text = state_text[len_welcome:]

    prompt=instruction+'\n'+state_text+'\n'+Action_prompt
    # print('ppp',prompt)
    # outputs = llm_policy.get_actions(prompt)
    # actions, hiddens = outputs
    if alg=='dqn':
        actions=adjust_list_length(admissible_commands)
    else:
        actions=get_thought_actions(llm_policy,llm_thought_policy,instruction,state_text,Action_prompt)


    # print("xxx",actions)
    # detected_action_idx = []
    detected_action = []
    feat_list = []
    for i, a in enumerate(actions):
        _decoded_a = process_text2action(a,admissible_commands)
        # detected_action_idx.append(_decoded_a)
        detected_action.append(_decoded_a)
        if not action_normalize:
            feat_list.append(f"{state_text}, {a}")
    if action_normalize:
        current_feat = feat_generate(state_text, embed_tkn, embed_policy, detected_action, device,task)
    else:
        _feat_encoded = embed_tkn(feat_list, return_tensors='pt', padding=True, max_length=30)
        feat_encoded = {k: v.to(device) for k, v in _feat_encoded.items()}
        feat_output = embed_policy(**feat_encoded, output_hidden_states=True)
        current_feat = feat_output.hidden_states[-1].mean(-2)
    steps=0

    qprob = Q(current_feat)
    qprob_dist = torch.distributions.Categorical(logits=qprob.squeeze(-1))
    # sample
    p = random.uniform(0, 1)

    # 如果p < 0.3，则随机选择一个action
    if p < epsilon:
        sample_action_idx = random.choice(range(len(actions)))
        sample_action_idx=torch.tensor(sample_action_idx)
    else:
        # 否则选择Q值最大的action
        sample_action_idx = qprob.argmax()

    # sample_action_idx = int(qprob_dist.sample())
    sample_action_text = detected_action[sample_action_idx]
    current_buffer=[]
    while not done:
        steps+=1

        chosen_action=sample_action_text
        expect_action=info['expert_action']
        aux=0
        if chosen_action==expect_action:
            aux=0.1
        observation, reward, done, truncated, info = env.step(chosen_action)
        next_state_text=observation['observation']
        # reward=reward*5+aux

        # len_welcome = len('-= Welcome to TextWorld, ALFRED! =-\n\n')
        if '-= Welcome to TextWorld, ALFRED! =-\n\n' in next_state_text:
            next_state_text = next_state_text[len_welcome:]

        # next_prompt=instruction+'\n' +'Histroys:'+'\n'+ '\n'.join(historys)+ 'Current obsevation:'+next_state_text+'\n'+Action_prompt
        next_prompt=instruction+'\n' + 'Current obsevation:'+next_state_text+'\n'+Action_prompt


        next_admissible_commands=info['admissible_commands']
        t_example=(instruction,task,state_text, chosen_action,next_state_text,reward,done,admissible_commands,next_admissible_commands)
        trajectory.append(t_example)
        total_reward += reward

        #
        # next_outputs = llm_policy.get_actions(next_prompt)
        #
        # next_actions, next_hiddens = next_outputs
        if alg=='dqn':
            next_actions = adjust_list_length(next_admissible_commands)
        else:
            next_actions = get_thought_actions(llm_policy, llm_thought_policy, instruction, next_state_text, Action_prompt,'\n'.join(historys))

        next_detected_action = []
        next_feat_list = []
        for ii,na in enumerate(next_actions):
            _next_decoded_a = process_text2action(na,next_admissible_commands)
            # next_detected_action_idx.append(_next_decoded_a)
            next_detected_action.append(_next_decoded_a)
            if not action_normalize:
                next_feat_list.append(f"{next_state_text}, {na}")

        if action_normalize:
            next_feat = feat_generate(next_state_text, embed_tkn, embed_policy, next_detected_action, device,task)
        else:
            _next_feat_encoded = embed_tkn(next_feat_list, return_tensors='pt', padding=True, max_length=30)
            next_feat_encoded = {k: v.to(device) for k, v in _next_feat_encoded.items()}
            next_feat_output = embed_policy(**next_feat_encoded, output_hidden_states=True)
            next_feat = next_feat_output.hidden_states[-1].mean(-2)

            #print(f"feat = {next_feat_list}, feat dim = {next_feat.shape}")

        # next_feat_all_list = []


        if len(historys)>=hislen:
            historys.pop(0)
        historys.append(f'Observation: {extract_before_phrase_regex(state_text, phrase)}. Action: {sample_action_text}.\n')

        next_qprob = Q(next_feat)
        # print(next_qprob)
        next_qprob_dist = torch.distributions.Categorical(logits=next_qprob.squeeze(-1))
        # sample
        # next_sample_action_idx = next_qprob.argmax()
        p = random.uniform(0, 1)

        if p < epsilon:
            next_sample_action_idx = random.choice(range(len(actions)))
            next_sample_action_idx = torch.tensor(next_sample_action_idx)
        else:
            # 否则选择Q值最大的action
            next_sample_action_idx = next_qprob.argmax()

        next_sample_action_text = next_detected_action[next_sample_action_idx]

        # print(current_feat.size())
        _data_tuple = {"feat": current_feat[sample_action_idx,:].unsqueeze(0).detach().cpu().numpy(),
                       "feat_all": current_feat.detach().cpu().numpy(),
                       # "action": F.one_hot(torch.tensor(sample_action_idx), num_classes=current_feat.shape[0]).cpu().numpy(),
                       # "action": F.one_hot(sample_action_idx.clone().detach(),
                       #                     num_classes=current_feat.shape[0]).cpu().numpy(),
                       "reward": np.array(reward,dtype=np.float32),
                       'done': np.array(done),
                       'next_feat': next_feat.to(torch.float32).detach().cpu().numpy(),
                       }
        current_buffer.append(_data_tuple)
        # if len(buffer) > capacity:
        #     buffer.pop(0)
        # buffer.append(_data_tuple)

        current_feat = next_feat
        state_text = next_state_text
        detected_action = next_detected_action
        admissible_commands=next_admissible_commands
        sample_action_text=next_sample_action_text
        sample_action_idx=next_sample_action_idx
        # torch.cuda.empty_cache()
        if steps>60:
            break
    if task not in example_dict:
        example_dict[str(task)]=(0,0)
    good_example,bad_example=example_dict[str(task)]
    if (total_reward == 1) or bad_example_t <= good_example_t or alg=='dqn':  # (total_reward==1 and good_example-bad_example<=10) or (total_reward==0 and bad_example-good_example<=10):#total_reward==1 or bad_example<=good_example:# or random.random()<0.2:
        if total_reward == 1:
            good_example += 1  # len(current_buffer)
            good_example_t += 1
        else:
            bad_example += 1  # len(current_buffer)
            bad_example_t += 1
        for i in range(len(current_buffer)):
            if len(buffer) > capacity:
                buffer.pop(0)
            buffer.append(current_buffer[i])
        # while alg=='dqn' and good_example_t
        trajectories_bala.append((steps,total_reward,trajectory))
    while alg=='dqn' and good_example_t<bad_example_t:
        for i in range(len(current_buffer)):
            if len(buffer) > capacity:
                buffer.pop(0)
            buffer.append(current_buffer[i])
        good_example_t+=1
    example_dict[str(task)] = (good_example, bad_example)
    print(steps,total_reward)
    trajectories.append((steps,total_reward,trajectory))
    # print("examples:",good_example,bad_example)
    return buffer,total_reward



def evaluate_q(env, embed_tkn, embed_policy, Q,
                      llm_policy,llm_thought_policy, device, action_normalize,alg):
    # action_choices = ['left', 'down', 'right', 'up']
    global test_env_ind
    total_reward = 0
    env_ret,info = env.reset()
    Action_prompt='Please select an action from the admissible actions. \n Action:'

    #obs, prompt, state_text = env_ret['obs'], env_ret['text'], env_ret['state_text']

    instruction=env_ret['instruction']
    task=extract_task(instruction)
    target_task = test_envs[test_env_ind]
    while target_task not in task:
        env_ret, info = env.reset()
        # obs, prompt, state_text = env_ret['obs'], env_ret['text'], env_ret['state_text']

        instruction = env_ret['instruction']
        task = extract_task(instruction)
    print(task)
    test_env_ind += 1
    if test_env_ind == len(test_envs):
        test_env_ind = 0
    admissible_commands=info['admissible_commands']
    state_text = env_ret['observation']
    len_welcome = len('-= Welcome to TextWorld, ALFRED! =-\n\n')
    if '-= Welcome to TextWorld, ALFRED! =-\n\n' in state_text:
        state_text = state_text[len_welcome:]

    prompt=instruction+'\n'+state_text+'\n'+Action_prompt
    admissible_commands=info['admissible_commands']

    done = False
    historys=[]
    step=0
    while not done:
        step+=1
        if alg=='dqn':
            actions=adjust_list_length(admissible_commands)
        else:
            actions = get_thought_actions(llm_policy, llm_thought_policy, instruction, state_text, Action_prompt,'\n'.join(historys))

        detected_action = []
        feat_list = []
        for i, a in enumerate(actions):
            _decoded_a = process_text2action(a,admissible_commands)
            detected_action.append(_decoded_a)
            if not action_normalize:
                feat_list.append(f"{state_text}, {a}")

        if action_normalize:
            current_feat = feat_generate(state_text, embed_tkn, embed_policy, detected_action, device,task)
        else:
            _feat_encoded = embed_tkn(feat_list, return_tensors='pt', padding=True, max_length=30)
            feat_encoded = {k: v.to(device) for k, v in _feat_encoded.items()}
            feat_output = embed_policy(**feat_encoded, output_hidden_states=True)
            current_feat = feat_output.hidden_states[-1].mean(-2)


        qprob = Q(current_feat)
        max_qprob = qprob.argmax()
        best_action = detected_action[max_qprob]

        if len(historys)>=hislen:
            historys.pop(0)

        historys.append(f'Observation: {extract_before_phrase_regex(state_text, phrase)}. Action: {best_action}.\n')

        # chosen_action_idx = process_text2action(best_action)

        observation, reward, done, truncated, info = env.step(best_action)
        next_state_text = observation['observation']

        if '-= Welcome to TextWorld, ALFRED! =-\n\n' in next_state_text:
            next_state_text = next_state_text[len_welcome:]


        next_prompt=instruction+'\n' + 'Current obsevation:'+next_state_text+'\n'+Action_prompt


        next_admissible_commands = info['admissible_commands']
        total_reward += reward


        prompt = next_prompt
        state_text = next_state_text
        admissible_commands=next_admissible_commands
        if step>60:
            break
    print(task,step,total_reward)
    return total_reward

from collections import Counter


def convert_train_data(buffer):

    data_dict = dict(zip(['feat','reward', 'done', "next_feat","feat_all"],
                         [
                          [i['feat'] for i in buffer],
                             # [i['action'] for i in buffer],
                          [i['reward'] for i in buffer],
                          [i['done'] for i in buffer],
                          [i['next_feat'] for i in buffer],
                          [i['feat_all'] for i in buffer],

    ]))
    raw_data_dict = datasets.Dataset.from_dict(data_dict)

    return raw_data_dict


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--capacity", type=int, default=4000)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--target_update", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_iter", type=int, default=50)
    parser.add_argument("--print_interval", type=int, default=25)
    parser.add_argument("--update_tau", type=float, default=0.005)
    parser.add_argument("--cql_coef", type=float, default=0.1)
    parser.add_argument("--update_type", type=str, default="soft")
    parser.add_argument("--alg", type=str, default="dqn")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--epsilon", type=float, default=0.1)

    parser.add_argument("--num_gens", type=int, default=5)
    parser.add_argument("--gen_batch_size", type=int, default=5)
    parser.add_argument("--thought_num_gens", type=int, default=1)
    parser.add_argument("--thought_gen_batch_size", type=int, default=1)

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--action_normalize", type=bool, default=False)


    parser.add_argument("--reference_llm_path", type=str, default='/xxx/llms/qwen/Qwen1___5-7B/')


    parser.add_argument("--embedding_llm_path", type=str, default='/xxx/llms/bert-base-uncased')

    parser.add_argument("--checkpoint_path", type=str, default='checkpoints')
    args = parser.parse_args()
    seed=args.seed

    random.seed(seed)

    # 设置 NumPy 中的随机种子
    np.random.seed(seed)

    # 设置 Torch 中的随机种子
    torch.manual_seed(seed)

    # 如果使用了 GPU，还需要设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config_path = os.path.join(ROOT_PATH, "config/aif_thought_config.yaml")
    cfg = OmegaConf.load(config_path)
    # env = FrozenLakeEnv(cfg.env_cfg)
    env = gym.make('llf-alfworld', instruction_type='b', feedback_type=['r','hp','hn'])

    replay_buffer= []
    epoch = args.epoch
    SAVE = False
    CAPACITY = args.capacity
    # embedding_dim = 768
    eval_interval = args.eval_interval
    target_update_interval = args.target_update
    device = f"cuda:{args.device}"
    model_path = args.reference_llm_path
    if args.alg=='dqn':
        good_example_t=0
    # LLM

    if 'Llama' in model_path:
        thought_policy = transformers.LlamaForCausalLM.from_pretrained(model_path,
                                                                           low_cpu_mem_usage=True,
                                                                           device_map=device,
                                                                           torch_dtype=torch.bfloat16)
        thought_tkn = transformers.AutoTokenizer.from_pretrained(model_path)


    else:
        thought_policy = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                                       low_cpu_mem_usage=True,
                                                                       device_map=device,
                                                                       torch_dtype=torch.bfloat16)
        thought_tkn = transformers.AutoTokenizer.from_pretrained(model_path)
        # from peft import PeftModel
        # thought_policy = PeftModel.from_pretrained(thought_policy, model_path)
        config = LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
        # thought_policy = get_peft_model(thought_policy, config)#, adapter_name="A")
        # thought_policy.load_adapter("/mnt/nasdata/yanxue/PromptBilevel/Alfworldsftqwen2tk"+'/A', adapter_name ='A')
        #
        # thought_policy.eval()

    if thought_tkn.pad_token_id is None:
        thought_tkn.pad_token_id = thought_tkn.eos_token_id
    llm = LLM_Generate_Instance(cfg,
                                thought_tkn,
                                thought_policy,
                                num_gens=args.num_gens,
                                gen_bz=args.gen_batch_size,
                                device=device,
                                stop_words=['<END>','<eos>'],
                                max_new_tokens=10)
    llm_thought = LLM_Generate_Instance(cfg,
                                thought_tkn,
                                thought_policy,
                                num_gens=args.thought_num_gens,
                                gen_bz=args.thought_gen_batch_size,
                                device=device,
                                        stop_words=['<END>','<eos>'],
                                        max_new_tokens=100)
    embedding_tkn = transformers.AutoTokenizer.from_pretrained(args.embedding_llm_path)
    embedding_policy = transformers.AutoModelForTokenClassification.from_pretrained(args.embedding_llm_path, device_map={"":device})
    # embedding_tkn=thought_tkn
    # embedding_policy=thought_policy
    if args.alg=='dqn':
        llm=None
        llm_thought=None
        print('xxxxxxxxxxxxxxxx')

    # Q prob
    embedding_dim = embedding_policy.config.hidden_size
    prob_network = LinearProb(embedding_dim).to(device)

    target_prob_network = LinearProb(embedding_dim).to(device)
    soft_update(target_prob_network, prob_network, tau=1)
    prob_optimizer = torch.optim.AdamW(prob_network.parameters(), lr=args.lr)


    current_train_step = 0
    save_interval = 5000
    total_loss_list = []
    total_eval_list = []
    total_step_cnt = 0
    batch_size = args.batch_size
    training_iter = args.train_iter
    current_best_r = -1

    model_save_path = os.path.join(ROOT_PATH, f"{args.checkpoint_path}/QProb_{args.seed}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    test_performace=[]
    train_reward_list=[]

    for iter in range(epoch):
        if ((iter)%10==0):
            print(len(example_dict.keys()))
            print(example_dict.keys())
            print(example_dict)
        if (iter)%eval_interval==0:
            eval_r = []
            for _ in range(len(test_envs)):
                r = evaluate_q(env, embedding_tkn, embedding_policy, prob_network, llm,llm_thought, device,
                        args.action_normalize,args.alg)
                eval_r.append(r)
            print(f"Epoch {iter}, Eval R = {np.mean(eval_r)}")
            total_eval_list.append(np.mean(eval_r))
            torch.save(total_eval_list, os.path.join(model_save_path, f"eval_r.pt"))

            if np.mean(eval_r) >= current_best_r:
                current_best_r = np.mean(eval_r)
                torch.save(prob_network, os.path.join(model_save_path, f"qprob_current_best.pt"))
                print(f"\n Model Saved !!! \n")


        replay_buffer,train_reward = explore_rollout_q(env, embedding_tkn, embedding_policy,
                                          prob_network, llm,llm_thought, replay_buffer, CAPACITY, device, args.action_normalize,args.epsilon,args.alg)

        train_reward_list.append(train_reward)
        if (iter+1)%10==0:
            torch.save(train_reward_list, os.path.join(model_save_path, f"train_r.pt"))

        if len(replay_buffer) <= batch_size or iter<40 or iter%10!=0:#20 change to 40 for reuse
            continue
        # if iter%10!=0:
        #     continue
        # dataset

        train_data = convert_train_data(replay_buffer)
        sampler = RandomSampler(train_data, replacement=True, num_samples=4*(len(replay_buffer)))#training_iter * batch_size)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, collate_fn=default_data_collator)

        #training
        batch_loss = []
        batch_loss_mse = []
        batch_loss_cql = []

        for step, batch in enumerate(train_dataloader):
            current_feat_batch = batch['feat'].float().to(device)              #[batch size, num_action, feat_dim]
            next_feat_batch = batch['next_feat'].float().to(device)           #[batch size, num_action, feat_dim]
            current_feat_all_batch = batch['feat_all'].float().to(device)  # [batch size, num_action, feat_dim]


            p = prob_network(current_feat_batch).squeeze(-1) #* action_batch     #[batch size, num_action]

            p_possible = prob_network(current_feat_all_batch).squeeze(-1)
            logsumexp = torch.logsumexp(p_possible, dim=1, keepdim=True)
            # print("log sum size", logsumexp.size())

            cql_loss = (logsumexp - p).mean()

            # print('ffffff',next_feat_batch.shape)
            # action_batch = batch['action'].float().to(device)                  #[batch size, num_action]
            reward_batch = batch['reward'].float().to(device)                 #[batch size, num_gen]
            done_batch = batch['done'].float().to(device)

            #p = p.sum(-1, keepdim=True)
            with torch.no_grad():
                target_p = target_prob_network(next_feat_batch).squeeze(-1)     #[batch size, num_action]
                greedy_target_p = target_p.max(-1)[0]
                target_value = reward_batch + 0.99*(1-done_batch)*greedy_target_p.unsqueeze(-1)

            mse_loss = torch.nn.functional.mse_loss(p, target_value.detach(), reduction='sum')
            loss = mse_loss#0.5 * mse_loss + args.cql_coef * cql_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(prob_network.parameters(), 1.0)
            prob_optimizer.step()
            prob_optimizer.zero_grad()

            current_train_step += 1

            if args.update_type=="soft":
                soft_update(target_prob_network, prob_network, tau=args.update_tau)
            else:
                if current_train_step % target_update_interval == 0:
                    soft_update(target_prob_network, prob_network, tau=1.0)

            batch_loss.append(0.5*mse_loss.item()+ args.cql_coef * cql_loss.item())
            batch_loss_mse.append(mse_loss.item())
            batch_loss_cql.append(cql_loss.item())

        if iter%args.print_interval==0:
            # print(f"Epoch {iter}, Loss = {np.mean(batch_loss)}")
            print(f"Epoch {iter}, Loss = {np.mean(batch_loss), np.mean(batch_loss_mse), np.mean(batch_loss_cql)}")
            total_loss_list.append([np.mean(batch_loss), np.mean(batch_loss_mse), np.mean(batch_loss_cql)])
            torch.save(total_loss_list, os.path.join(model_save_path, f"loss.pt"))

            # total_loss_list.append(np.mean(batch_loss))

        # eval
    print(f'Eval = {total_eval_list}')

    import pickle



    filename =  os.path.join(model_save_path, f"raw_trajectory_data.pt")
    #
    # # 将列表写入 pickle 文件
    with open(filename, 'wb') as file:
        pickle.dump(trajectories, file)

    torch.save(replay_buffer, os.path.join(model_save_path, f"offline_buffer_bala.pth"))

