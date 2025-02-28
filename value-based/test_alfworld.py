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
hard_tasks=['put some cellphone on sidetable.', 'put a cellphone in bed.', 'put some cd on dresser.', 'put some cellphone on sidetable.', 'put a cellphone in sidetable.', 'put some mug on sidetable.', 'put a fork in drawer.', 'put some fork on drawer.', 'put a cellphone in desk.', 'put a fork in drawer.', 'put a cellphone in desk.', 'put some fork on drawer.', 'put some mug on sidetable.', 'put some cellphone on sidetable.', 'put a cellphone in sidetable.', 'put some cellphone on sidetable.', 'put a cellphone in bed.', 'put some cd on dresser.']
hard_tasks=['put some cellphone on sidetable.', 'put a cellphone in bed.', 'put some fork on drawer.', 'put a cellphone in sidetable.', 'put some cd on dresser.']

test_envs=['Task: put a soapbottle in toilet.', 'Task: put a handtowel in garbagecan.', 'Task: put some cellphone on sidetable.', 'Task: put some candle on toilet.', 'Task: put a cellphone in sidetable.', 'Task: put some soapbottle on toilet.', 'Task: put a candle in toilet.']
test_envs=['put some cellphone on armchair.', 'put some candle on sidetable.', 'put some candle on toilet.', 'put a cellphone in armchair.', 'put some soapbottle on toilet.', 'put some cellphone on bed.', 'put some cellphone on sidetable.', 'put a cloth in toilet.', 'put some cellphone on desk.', 'put a cellphone in drawer.', 'put some spraybottle on countertop.', 'put some pillow on sofa.', 'put some soapbottle on cart.', 'put some spraybottle on toilet.', 'put some laptop on armchair.', 'put a candle in sidetable.', 'put a cellphone in bed.', 'put a handtowel in garbagecan.', 'put some cd on dresser.', 'put some box on armchair.', 'put a cd in dresser.', 'put a spraybottle in countertop.', 'put a candle in toilet.', 'put a soapbottle in toilet.', 'put a laptop in armchair.', 'put a cellphone in sidetable.', 'put some mug on sidetable.', 'put a fork in drawer.', 'put a pillow in sofa.', 'put some fork on drawer.', 'put a cellphone in desk.']

test_envs = [task for task in test_envs if task not in hard_tasks]



print(test_envs)

# test_envs=[
#     'put a pillow in sofa.', 'put a cellphone in desk.']

env_ind=0
test_env_ind=0
import numpy as np

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
    def __init__(self, config, tkn, policy,  num_gens, gen_bz, device,temperature=1.0,top_p=1.0,stop_words=["\n"],max_new_tokens=30):
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
        self.temperature =temperature#1.0# 0.8#1.0#0.8#1.0#1.0#1.0#0.8#0.1#1.0#0.8#0.8
        self.batch_size = gen_bz           # splitted into how many batches with this batch size
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p#1.0#0.95#1.0#1.0#1.0#1.0#0.75#0.95#0.8#0.95#0.8#0.95
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
    return actions_
example_dict={}
good_example_t=0
bad_example_t=0
def convert_to_tenser_dataset(data_list,embed_tkn, embed_policy,llm_policy, buffer,
                      capacity, device, action_normalize):
    acc=0
    for example in data_list:
        instruction,task,state,action,next_state_text,reward,done,admissible_commands,next_admissible_commands=example
        state_text=state
        detected_action=[action]
        if action_normalize:
            current_feat = feat_generate(state_text, embed_tkn, embed_policy, detected_action, device, task)

        ##current sample
        actions = get_thought_actions(llm_policy, '', instruction, state_text, Action_prompt,'')
        detected_action = []
        for ii, na in enumerate(actions):
            _decoded_a = process_text2action(na, admissible_commands)
            # next_detected_action_idx.append(_next_decoded_a)
            detected_action.append(_decoded_a)

        if action_normalize:
            current_feat_all = feat_generate(state_text, embed_tkn, embed_policy, detected_action, device, task)
        if action in detected_action:
            acc+=1
        #
        next_actions = get_thought_actions(llm_policy, '', instruction, next_state_text, Action_prompt,'')

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

        _data_tuple = {"feat": current_feat.detach().cpu().numpy(),
                       "feat_all": current_feat_all.detach().cpu().numpy(),
                       # "action": F.one_hot(torch.tensor(sample_action_idx), num_classes=current_feat.shape[0]).cpu().numpy(),
                       "reward": np.array(reward, dtype=np.float32),
                       'done': np.array(done),
                       'next_feat': next_feat.to(torch.float32).detach().cpu().numpy(),
                       }
        buffer.append(_data_tuple)
    print(len(buffer),acc)
    return buffer

def evaluate_q(env, embed_tkn, embed_policy, Q,
                      llm_policy,llm_thought_policy, device, action_normalize,Q_2):
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
    observation = env_ret['observation']

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
        # qprob_2=Q_2(current_feat)
        max_qprob = qprob.argmax()
        best_action = detected_action[max_qprob]

        if len(historys)>=hislen:
            historys.pop(0)

        historys.append(f'Observation: {extract_before_phrase_regex(state_text, phrase)}. Action: {best_action}.\n')

        try:

            observation, reward, done, truncated, info = env.step(best_action)
        except Exception:
            print('wrong',best_action)
            pass

        next_state_text = observation['observation']

        if '-= Welcome to TextWorld, ALFRED! =-\n\n' in next_state_text:
            next_state_text = next_state_text[len_welcome:]


        next_prompt=instruction+'\n' + 'Current obsevation:'+next_state_text+'\n'+Action_prompt


        next_admissible_commands = info['admissible_commands']
        # next_obs, next_prompt, next_state_text, reward, done = list(next_env_ret.values())
        total_reward += reward


        prompt = next_prompt
        state_text = next_state_text
        admissible_commands=next_admissible_commands
        torch.cuda.empty_cache()
        if step>60:
            break
    print(task,step,total_reward)
    return total_reward



def convert_train_data(buffer):

    data_dict = dict(zip(['feat','reward', 'done', "next_feat","feat_all"],
                         [
                          [i['feat'] for i in buffer],
                          [i['reward'] for i in buffer],
                          [i['done'] for i in buffer],
                          [i['next_feat'] for i in buffer],
                             [i['feat_all'] for i in buffer],

                         ]))
    raw_data_dict = datasets.Dataset.from_dict(data_dict)

    return raw_data_dict


if __name__ == '__main__':
    random.seed(42)

    # 设置 NumPy 中的随机种子
    np.random.seed(42)

    # 设置 Torch 中的随机种子
    torch.manual_seed(42)

    # 如果使用了 GPU，还需要设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
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
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--num_gens", type=int, default=5)
    parser.add_argument("--gen_batch_size", type=int, default=5)
    parser.add_argument("--thought_num_gens", type=int, default=1)
    parser.add_argument("--thought_gen_batch_size", type=int, default=1)

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--action_normalize", type=bool, default=False)


    parser.add_argument("--reference_llm_path", type=str, default='/xxx/llms/qwen/Qwen1___5-32B/')
    parser.add_argument("--saved_qnetwork", type=str, default='xxx/qprob_current_best.pt')


    parser.add_argument("--embedding_llm_path", type=str, default='/xxx/llms/bert-base-uncased')

    parser.add_argument("--checkpoint_path", type=str, default='checkpoints')
    args = parser.parse_args()


    config_path = os.path.join(ROOT_PATH, "config/aif_thought_config.yaml")
    cfg = OmegaConf.load(config_path)
    # env = FrozenLakeEnv(cfg.env_cfg)
    env = gym.make('llf-alfworld', instruction_type='b', feedback_type=['r','hp','hn'])

    replay_buffer= []
    epoch = args.epoch
    cql_coef = args.cql_coef
    SAVE = False
    CAPACITY = args.capacity
    # embedding_dim = 768
    eval_interval = args.eval_interval
    target_update_interval = args.target_update
    device = f"cuda:{args.device}"

    # LLM
    model_path = args.reference_llm_path
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
                                temperature=args.temperature,
                                top_p=args.top_p,
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

    # Q prob
    embedding_dim = embedding_policy.config.hidden_size
    prob_network = LinearProb(embedding_dim).to(device)
    prob_network = torch.load(f'/xxx/{args.saved_qnetwork}').to(device)

    target_prob_network = LinearProb(embedding_dim).to(device)
    soft_update(target_prob_network, prob_network, tau=1)

    prob_network_2 = LinearProb(embedding_dim).to(device)

    target_prob_network_2 = LinearProb(embedding_dim).to(device)
    soft_update(target_prob_network_2, prob_network_2, tau=1)

    prob_optimizer = torch.optim.AdamW(prob_network.parameters(), lr=args.lr)
    prob_optimizer_2 = torch.optim.AdamW(prob_network_2.parameters(), lr=args.lr)


    current_train_step = 0
    save_interval = 5000
    total_loss_list = []
    total_eval_list = []
    total_step_cnt = 0
    batch_size = args.batch_size
    training_iter = args.train_iter
    current_best_r = -1
    eval_rewards=[]
    model_save_path = os.path.join(ROOT_PATH, f"{args.checkpoint_path}/QProb")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    test_performace=[]
    train_reward_list=[]

    train_data = convert_train_data(replay_buffer)
    sampler = RandomSampler(train_data, replacement=True,num_samples=training_iter * batch_size)
                            #num_samples=4 * (len(replay_buffer)))  # training_iter * batch_size)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, collate_fn=default_data_collator)

    # training
    batch_loss = []
    batch_loss_mse = []
    batch_loss_cql = []
    batch_loss_2 = []
    batch_loss_mse_2 = []
    batch_loss_cql_2 = []
    eval_r = []
    for _ in range(len(test_envs)):
        r = evaluate_q(env, embedding_tkn, embedding_policy, prob_network, llm, llm_thought, device,
                       args.action_normalize, prob_network_2)
        eval_r.append(r)
    print(f"Epoch {iter}, Eval R = {np.mean(eval_r)}")
    print(eval_r)
