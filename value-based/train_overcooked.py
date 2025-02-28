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
# import llfbench as gym
import concurrent.futures

from torch.utils.data import DataLoader
from transformers import default_data_collator,LlamaForCausalLM, LlamaTokenizer
from AIF.env.frozen_lake import FrozenLakeEnv
from torch.utils.data import RandomSampler
from utils import obs2text
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

import json
from datetime import datetime
import os
from pathlib import Path
import time
print('1111111')
hislen=3
ROOT_PATH = str(Path(__file__).resolve().parent)
task=0
import gym

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
        self.temperature = 1.0#0.8#1.0#0.8#0.8#0.8
        self.batch_size = gen_bz           # splitted into how many batches with this batch size
        self.max_new_tokens = max_new_tokens
        self.top_p = 1.0#0.95#1.0#0.95#0.8#0.95
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



def feat_generate(state_text, tkn, p, gen_action, device,task):
        phrase = "You are allowed to take the following actions:"

    # action_choices = ['left', 'down', 'right', 'up']
        feat_list = []
        state_text = extract_before_phrase_regex(state_text, phrase)
        for i, a in enumerate(gen_action):
            input_action = a
            state_info = f"Observation: {state_text},  Action: {input_action}."
            # print(state_info)
            feat_list.append(state_info)
        _feat_encoded = tkn(feat_list, return_tensors='pt', padding=True, max_length=40)
        feat_encoded = {k: v.to(device) for k, v in _feat_encoded.items()}
        feat_output = p(**feat_encoded, output_hidden_states=True)
        feat = feat_output.hidden_states[-1][:,0,:]#.mean(-2)
        return feat

def adjust_list_length(lst, desired_length=10, fill_value="nope"):
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

saved_actions={}
query_time=0
def get_thought_actions(llm_policy,llm_thought_policy,instruction,state_text,Action_prompt,history=''):
    global saved_actions,query_time
    # actions_, _ = llm_policy.get_actions(instruction+'\n'+'Current observation:'+state_text+'\n'+Action_prompt)
    # print(actions_)
    # history=''
    text_obs = 'History:\n'+history+'\n'+"Current Observation: "+state_text
    # print(instruction+'\n'+'History:\n'+history+'\n'+"Current Observation: "+state_text+'\n'+Action_prompt)
    if text_obs in saved_actions.keys():
        actions_ = saved_actions[text_obs]
    else:
        actions_, _ = llm_policy.get_actions(instruction+'\n'+'History:\n'+history+'\n'+"Current Observation: "+state_text+'\n'+Action_prompt)
        query_time += 1
        saved_actions[text_obs] = actions_
    return actions_
example_dict={}
good_example_t=10
bad_example_t=0
maximum_return=0
trajectories_bala=[]

def explore_rollout_q(env, embed_tkn, embed_policy, Q,
                      llm_policy,llm_thought_policy, buffer,
                      capacity, device, action_normalize,epsilon=0.0,alg='dqn',alpha=10):
    global example_dict,good_example_t,bad_example_t,env_ind,trajectories,trajectories_bala,task
    trajectory=[]
    vector_obs = env.reset()

    text_info = obs2text(vector_obs,task)
    state_text = text_info['prompt']
    state_text = extract_obs(state_text)
    admissible_commands = text_info['action']
    Action_prompt="To serve the dish of a bowl containing only chopped tomatoes, what action should you take next? Based on your previous observations and actions, please select only one action for the next step from the admissible actions: {}. Action:"
    #'To serve the dish of a bowl only containing chopped tomato, which action you should take for the next one step? Please select only one action for the next one step from the admissible actions: {}. \n Action:'

    instruction="Your task is to serve the dish of a bowl only containing chopped tomato."
    done = False
    historys=[]
    step=0

    if 'dqn' in alg:
        actions=adjust_list_length(admissible_commands)
    else:

        actions=get_thought_actions(llm_policy,llm_thought_policy,instruction,state_text,Action_prompt.format(admissible_commands))


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


    if 'dqn' in alg:
        if p < epsilon:
            sample_action_idx = random.choice(range(len(actions)))
            sample_action_idx = torch.tensor(sample_action_idx)
        else:
            # 否则选择Q值最大的action
            sample_action_idx = qprob.argmax()
    else:
        qprob_dist = torch.distributions.Categorical(logits=alpha*qprob.squeeze(-1))
        sample_action_idx = int(qprob_dist.sample())#qprob.argmax()

    sample_action_text = detected_action[sample_action_idx]
    action = admissible_commands.index(sample_action_text)
    total_reward=0
    current_buffer=[]
    while not done:
        steps+=1
        next_obs_vector, reward, done, info = env.step([action])
        # print(next_obs_vector)
        next_text_info = obs2text(next_obs_vector, task)
        next_state_text = next_text_info['prompt']
        next_state_text = extract_obs(next_state_text)
        next_admissible_commands = next_text_info['action']

        total_reward += reward
        if len(historys)>=hislen:
            historys.pop(0)

        historys.append(f'Observation: {state_text}. Action: {sample_action_text}.')



        t_example=(instruction,state_text, sample_action_text,next_state_text,reward,done,admissible_commands,next_admissible_commands)
        trajectory.append(t_example)

        #
        # next_outputs = llm_policy.get_actions(next_prompt)
        #
        # next_actions, next_hiddens = next_outputs
        if 'dqn' in alg:
            next_actions=adjust_list_length(next_admissible_commands)
        else:
            next_actions = get_thought_actions(llm_policy, llm_thought_policy, instruction, next_state_text, Action_prompt.format(next_admissible_commands),'\n'.join(historys))

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


        next_qprob = Q(next_feat)
        # print(next_qprob)
        next_qprob_dist = torch.distributions.Categorical(logits=next_qprob.squeeze(-1))
        # sample
        # next_sample_action_idx = next_qprob.argmax()
        p = random.uniform(0, 1)


        if 'dqn' in alg:
            if p < epsilon:
                next_sample_action_idx = random.choice(range(len(actions)))
                next_sample_action_idx = torch.tensor(next_sample_action_idx)
            else:
                # 否则选择Q值最大的action
                next_sample_action_idx = next_qprob.argmax()

        else:
            # 否则选择Q值最大的action
            next_qprob_dist = torch.distributions.Categorical(logits=alpha * next_qprob.squeeze(-1))

            next_sample_action_idx = int(next_qprob_dist.sample())  # next_qprob.argmax()

        next_sample_action_text = next_detected_action[next_sample_action_idx]
        next_action= next_admissible_commands.index(next_sample_action_text)
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

        current_feat = next_feat
        state_text = next_state_text
        admissible_commands=next_admissible_commands
        sample_action_text=next_sample_action_text
        sample_action_idx=next_sample_action_idx
        action=next_action
        if steps>30:
            break
    if (total_reward >0.5) or bad_example_t <= good_example_t or ('bala' not in alg):  # (total_reward==1 and good_example-bad_example<=10) or (total_reward==0 and bad_example-good_example<=10):#total_reward==1 or bad_example<=good_example:# or random.random()<0.2:
            if total_reward >0.5:
                good_example_t += 1
            else:
                bad_example_t += 1
            for i in range(len(current_buffer)):
                if len(buffer) > capacity:
                    buffer.pop(0)
                buffer.append(current_buffer[i])
            trajectories_bala.append((steps,total_reward,trajectory))
    while 'dqn' in alg and ('bala' in alg) and good_example_t<bad_example_t:
        for i in range(len(current_buffer)):
            if len(buffer) > capacity:
                buffer.pop(0)
            buffer.append(current_buffer[i])
        good_example_t+=1

    print(steps,total_reward,bad_example_t,good_example_t)
    trajectories.append((steps,total_reward,trajectory))
    # print("examples:",good_example,bad_example)
    return buffer,total_reward

def extract_obs(obs):

    # Input text

    # Regular expression to extract content between two phrases
    pattern = r"(.*?)To serve the dish of a bowl only containing chopped tomato"

    # Extract the matched content
    match = re.search(pattern, obs)

    # Output the extracted content
    if match:
        extracted_content = match.group(1).strip()
        return extracted_content
    else:
        print("No match found")


def evaluate_q(env, embed_tkn, embed_policy, Q,
                      llm_policy,llm_thought_policy, device, action_normalize,alg,alpha):
    # action_choices = ['left', 'down', 'right', 'up']
    global task
    total_reward = 0
    vector_obs = env.reset()
    # print(vector_obs)
    # print(vector_obs.type)

    text_info=obs2text(vector_obs,task)
    state_text=text_info['prompt']
    admissible_commands=text_info['action']
    # print(admissible_commands)
    Action_prompt="To serve the dish of a bowl containing only chopped tomatoes, what action should you take next? Based on your previous observations and actions, please select only one action for the next step from the admissible actions: {}. Action:"
    #'To serve the dish of a bowl only containing chopped tomato, which action you should take for the next one step? Please select only one action for the next one step from the admissible actions: {}. \n Action:'

    instruction="Your task is to serve the dish of a bowl only containing chopped tomato."
    done = False
    historys=[]
    step=0
    while not done:
        step+=1
        # decision
        # outputs = llm_policy.get_actions(prompt)
        # actions, hiddens = outputs
        state_text=extract_obs(state_text)
        # print(state_text)
        # print(admissible_commands)
        if 'dqn' in alg:
            actions=adjust_list_length(admissible_commands)
        else:
            actions = get_thought_actions(llm_policy, llm_thought_policy, instruction, state_text, Action_prompt.format(admissible_commands),'\n'.join(historys))

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
        # max_qprob = qprob.argmax()
        # best_action = detected_action[max_qprob]
        qprob_dist = torch.distributions.Categorical(logits=alpha*qprob.squeeze(-1))
        max_qprob = int(qprob_dist.sample())#qprob.argmax()
        best_action = detected_action[max_qprob]

        action = admissible_commands.index(best_action)

        # print(best_action)
        # print(action)
        # print()
        if len(historys)>=hislen:
            historys.pop(0)

        historys.append(f'Observation: {state_text}. Action: {best_action}.')

        # chosen_action_idx = process_text2action(best_action)
        next_obs_vector, reward, done, info = env.step([action])
        # print(next_obs_vector)
        next_text_info = obs2text(next_obs_vector, task)
        next_state_text = next_text_info['prompt']
        # next_state_text = extract_obs(next_state_text)
        next_admissible_commands = next_text_info['action']




        total_reward += reward


        state_text = next_state_text
        admissible_commands=next_admissible_commands
        if step>30:
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

    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_gens", type=int, default=1)
    parser.add_argument("--gen_batch_size", type=int, default=1)
    parser.add_argument("--thought_num_gens", type=int, default=1)
    parser.add_argument("--thought_gen_batch_size", type=int, default=1)

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=10)

    parser.add_argument("--action_normalize", type=bool, default=False)

    parser.add_argument("--reference_llm_path", type=str, default='/xxx/llms/qwen/Qwen1___5-7B/')


    parser.add_argument("--embedding_llm_path", type=str, default='/xxx/llms/bert-base-uncased')
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--checkpoint_path", type=str, default='checkpoints')



    parser.add_argument('--env-id', action='store', type=str, default='Overcooked-LLMA-v4', help='Domain name')
    parser.add_argument('--n-agent', action='store', type=int, default=1, help='Number of agents')
    parser.add_argument('--grid-dim', action='store', type=int, nargs=2, default=[7, 7], help='Grid world size')
    parser.add_argument('--task', action='store', type=int, default=0, help='The receipt agent cooks')
    parser.add_argument('--map-type', action='store', type=str, default="A", help='The type of map')
    parser.add_argument('--obs-radius', action='store', type=int, default=2, help='The radius of the agents')
    parser.add_argument('--env-reward', action='store', type=float, nargs=4, default=[0.2, 1, 0.1, 0.001],
                        help='The reward list of the env')
    parser.add_argument('--mode', action='store', type=str, default="vector",
                        help='The type of the observation(vector/image)')
    parser.add_argument('--debug', action='store', type=bool, default=False,
                        help='Whehter print the debug information and render')

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


    rewardList = {"subtask finished": args.env_reward[0], "correct delivery": args.env_reward[1],
                  "wrong delivery": -args.env_reward[2], "step penalty": -args.env_reward[3]}
    TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad",
                "lettuce-onion salad", "lettuce-onion-tomato salad"]
    env_params = {'grid_dim': args.grid_dim,
                  'task': TASKLIST[args.task],
                  'rewardList': rewardList,
                  'map_type': args.map_type,
                  'n_agent': args.n_agent,
                  'obs_radius': args.obs_radius,
                  'mode': args.mode,
                  'debug': args.debug
                  }
    env = gym.make(args.env_id, **env_params)
    env = MacEnvWrapper(env)
    # global task
    task=args.task
    # text_obs = obs2text(obs)#[obs2text(o,args.task) for o in obs]
    # print(text_obs)

    replay_buffer= []
    epoch = args.epoch
    SAVE = False
    CAPACITY = args.capacity
    # embedding_dim = 768
    eval_interval = args.eval_interval
    target_update_interval = args.target_update
    device = f"cuda:{args.device}"

    # LLM
    model_path = args.reference_llm_path #"/mnt/yansong/YS/pretrained/qwen/qwen/Qwen1___5-4B"
    if 'Llama' in model_path:
        thought_policy = transformers.LlamaForCausalLM.from_pretrained(model_path,
                                                                           low_cpu_mem_usage=True,
                                                                           device_map=device,
                                                                           torch_dtype=torch.bfloat16)
        thought_tkn = transformers.AutoTokenizer.from_pretrained(model_path)


    else:
        thought_policy = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                                       low_cpu_mem_usage=True,
                                                                       device_map='auto',
                                                                       torch_dtype=torch.bfloat16)
        thought_tkn = transformers.AutoTokenizer.from_pretrained(model_path)
        config = LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )

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
    embedding_tkn = transformers.AutoTokenizer.from_pretrained(args.embedding_llm_path) #"/mnt/yansong/YS/pretrained/bert-base-uncased")
    embedding_policy = transformers.AutoModelForTokenClassification.from_pretrained(args.embedding_llm_path, device_map={"":device})
    # embedding_tkn=thought_tkn
    # embedding_policy=thought_policy
    # embedding_policy = transformers.AutoModelForCausalLM.from_pretrained(args.embedding_llm_path, device_map={"":device})

    # Q prob
    embedding_dim = embedding_policy.config.hidden_size
    prob_network = LinearProb(embedding_dim).to(device)
    if args.load_checkpoint is not None:
        prob_network = torch.load(f'/xxx/{args.load_checkpoint}').to(device)

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
    st_time=time.time()

    for iter in range(epoch):
        if ((iter)%10==0):
            print(len(example_dict.keys()))
            print(example_dict.keys())
            print(example_dict)

        if (iter)%eval_interval==0:
            eval_r = []
            for _ in range(10):
                r = evaluate_q(env, embedding_tkn, embedding_policy, prob_network, llm,llm_thought, device,
                        args.action_normalize,args.alg,args.alpha)
                eval_r.append(r)
            print(f"Epoch {iter}, Eval R = {np.mean(eval_r)}")
            total_eval_list.append(np.mean(eval_r))
            torch.save(total_eval_list, os.path.join(model_save_path, f"eval_r.pt"))

            if np.mean(eval_r) >= current_best_r:
                current_best_r = np.mean(eval_r)
                torch.save(prob_network, os.path.join(model_save_path, f"qprob_current_best.pt"))
                print(f"\n Model Saved !!! \n")


        replay_buffer,train_reward = explore_rollout_q(env, embedding_tkn, embedding_policy,
                                          prob_network, llm,llm_thought, replay_buffer, CAPACITY, device, args.action_normalize,args.epsilon,args.alg,args.alpha)

        train_reward_list.append(train_reward)
        if (iter+1)%10==0:
            torch.save(train_reward_list, os.path.join(model_save_path, f"train_r.pt"))

        if len(replay_buffer) <= batch_size  or iter%5!=0:#20 change to 40 for reuse
            continue
        # if iter%10!=0:
        #     continue
        # dataset

        train_data = convert_train_data(replay_buffer)
        sampler = RandomSampler(train_data, replacement=True, num_samples=training_iter * batch_size)
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
    torch.save(replay_buffer, os.path.join(model_save_path, f"offline_buffer_bala.pth"))
    en_time=time.time()
    print('time:',en_time-st_time)
    print('query_time:',query_time)
    import pickle



    filename =  os.path.join(model_save_path, f"raw_trajectory_data.pt")
    #
    # # 将列表写入 pickle 文件
    with open(filename, 'wb') as file:
        pickle.dump(trajectories, file)
    filename =  os.path.join(model_save_path, f"raw_trajectory_data_bala.pt")
    #
    # # 将列表写入 pickle 文件
    with open(filename, 'wb') as file:
        pickle.dump(trajectories_bala, file)
    import sys
    print(sys.getsizeof(saved_actions)/(1024.0*1024.0))

