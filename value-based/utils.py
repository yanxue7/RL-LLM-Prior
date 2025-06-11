import os
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # bind to all interfaces and use an OS provided port
        return s.getsockname()[1] # return only the port number


def get_remote_file(remote_path, local_path=None):
    hostname, path = remote_path.split(':')
    local_hostname = socket.gethostname()
    if hostname == local_hostname or hostname == local_hostname[:local_hostname.find('.')]:
        return path
    
    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')    
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Copying {hostname}:{path} to {local_path}')
    os.system(f'scp {remote_path} {local_path}')
    return local_path


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"
    

def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ''):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
        print('*' * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    rank0_print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)

def obs2text(obs,task):
    if task == 3:
        obs = obs.tolist()
        action_list = [
            "pick up the tomato",
            "pick up the lettuce",
            "pick up the onion",
            "take the empty bowl",
            "walk to the first cutting board",
            "walk to the second cutting board",
            "serve nothing",
            "chop nothing",
        ]

        ingredient_in_ori_pos = [0, 0, 0, 0]
        ingredient = ["a tomato", "a lettuce", "an onion", "a bowl"]
        raw_ingredient = ["tomato", "lettuce", "onion", "bowl"]
        chopped = [False, False, False]
        ori_pos = [[0, 5], [1, 6], [2, 6], [6, 5]]
        sentences = ["There are two fixed cutting boards in the room."]

        item = []
        item_index = []
        agent_pos = obs[17:19]
        first_cutting_board_pos = [1, 0]
        second_cutting_board_pos = [2, 0]

        item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos,
                    "in_second_cutting_board": second_cutting_board_pos}
        overlay = {"in_agent": [], "in_first_cutting_board": [], "in_second_cutting_board": []}

        for i in range(4):
            pos = obs[3 * i: 3 * i + 2]
            if pos == ori_pos[i]:
                ingredient_in_ori_pos[i] == 1
                item.append(ingredient[i])
                item_index.append(i)

            if i < 3 and obs[3 * i + 2] == 3:
                chopped[i] = True

            for k in overlay.keys():
                if pos == item_pos[k]:
                    overlay[k].append(i)

                    if len(overlay[k]) > 1:
                        action_list[3] = "take the bowl"

        if len(item) == 1:
            template = "You notice {} on the table."
        elif len(item) == 2:
            template = "You notice {} and {} on the different tables."
        elif len(item) == 3:
            template = "You notice {}, {} and {} on the different tables."
        elif len(item) == 4:
            template = "You notice {}, {}, {} and {} on the different tables."

        if len(item) > 0:
            sentences.append(template.format(*item).capitalize())

        cutting_board_index = ["first", "second"]
        cutting_board_name = ["in_first_cutting_board", "in_second_cutting_board"]
        for cindex in range(2):
            if len(overlay[cutting_board_name[cindex]]) == 1:
                id = overlay[cutting_board_name[cindex]][0]
                template = "{} is on the {} cutting board."
                if id == 3:
                    sentences.append(template.format("a bowl", cutting_board_index[cindex]).capitalize())
                else:
                    if chopped[id]:
                        sentences.append(template.format("a chopped " + raw_ingredient[id],
                                                         cutting_board_index[cindex]).capitalize())
                    else:
                        sentences.append(template.format("an unchopped " + raw_ingredient[id],
                                                         cutting_board_index[cindex]).capitalize())
                    if agent_pos == [cindex + 1, 1]:
                        action_list[-1] = "chop the " + raw_ingredient[id]

            elif len(overlay[cutting_board_name[cindex]]) > 1:
                in_plate_item = overlay[cutting_board_name[cindex]][:-1]
                if len(in_plate_item) == 1:
                    full_plate_template = "A bowl containing chopped {} is on the {} cutting board."
                elif len(in_plate_item) == 2:
                    full_plate_template = "A bowl containing chopped {} and {} is on the {} cutting board."
                elif len(in_plate_item) == 3:
                    full_plate_template = "A bowl containing chopped {}, {} and {} is on the {} cutting board."
                sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item],
                                                            cutting_board_index[cindex]).capitalize())

                # in front of cutting board 1
        if agent_pos == [1, 1]:
            cindex = 0
        # in front of cutting board 2
        elif agent_pos == [2, 1]:
            cindex = 1
        else:
            cindex = -1

        action_template = "put the {} on the {} cutting board"
        hold_bowl_action = [
            "put the tomato in the bowl",
            "put the lettuce in the bowl",
            "put the onion in the bowl",
        ]

        if cindex >= 0:
            if len(overlay["in_agent"]) == 0:
                template = "Currently you are standing in front of the {} cutting board without anything in hand."
                sentences.append(template.format(cutting_board_index[cindex]).capitalize())

            elif len(overlay["in_agent"]) == 1:
                action_list[6] = "serve the dish"
                id = overlay["in_agent"][0]
                template = "Currently you are standing in front of the {} cutting board, carrying {} in hand."
                if id == 3:
                    sentences.append(template.format(cutting_board_index[cindex], "a bowl").capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format(raw_ingredient[id], "first")
                    action_list[5] = action_template.format(raw_ingredient[id], "second")
                else:
                    if chopped[id]:
                        sentences.append(template.format(cutting_board_index[cindex],
                                                         "a chopped " + raw_ingredient[id], ).capitalize())
                    else:
                        sentences.append(template.format(cutting_board_index[cindex],
                                                         "an unchopped " + raw_ingredient[id]).capitalize())
                        action_list[4] = action_template.format(raw_ingredient[id], "first")
                        action_list[5] = action_template.format(raw_ingredient[id], "second")
            elif len(overlay["in_agent"]) > 1:
                action_list[6] = "serve the dish"
                in_plate_item = overlay["in_agent"][:-1]
                if len(in_plate_item) == 1:
                    full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} in hand."
                elif len(in_plate_item) == 2:
                    full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} and {} in hand."
                elif len(in_plate_item) == 3:
                    full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {}, {} and {} in hand."

                sentences.append(full_plate_template.format(cutting_board_index[cindex],
                                                            *[raw_ingredient[id] for id in in_plate_item]).capitalize())
                action_list[:3] = hold_bowl_action
                action_list[4] = action_template.format("bowl", "first")
                action_list[5] = action_template.format("bowl", "second")
        else:
            if len(overlay["in_agent"]) == 0:
                template = "Currently you don't have anything in hand."
                sentences.append(template.format(cutting_board_index[cindex]).capitalize())

            elif len(overlay["in_agent"]) == 1:
                action_list[6] = "serve the dish"
                id = overlay["in_agent"][0]
                template = "Currently you are carrying {} in hand."
                if id == 3:
                    sentences.append(template.format("a bowl").capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format(raw_ingredient[id], "first")
                    action_list[5] = action_template.format(raw_ingredient[id], "second")
                else:
                    if chopped[id]:
                        sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize())
                    else:
                        sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                        action_list[4] = action_template.format(raw_ingredient[id], "first")
                        action_list[5] = action_template.format(raw_ingredient[id], "second")
            elif len(overlay["in_agent"]) > 1:
                action_list[6] = "serve the dish"
                in_plate_item = overlay["in_agent"][:-1]
                if len(in_plate_item) == 1:
                    full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                elif len(in_plate_item) == 2:
                    full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                elif len(in_plate_item) == 3:
                    full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."

                sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                action_list[:3] = hold_bowl_action
                action_list[4] = action_template.format("bowl", "first")
                action_list[5] = action_template.format("bowl", "second")
        sentences.append("To serve the dish of a bowl only containing chopped tomato and lettuce, you should first")
    elif task == 0:
        obs = obs.tolist()

        action_list = [
            "pick up the tomato",
            "take the bowl",
            "walk to the cutting board",
            "serve nothing",
            "chop nothing",
        ]

        ingredient_in_ori_pos = [0, 0]
        ingredient = ["a tomato", "a bowl"]
        raw_ingredient = ["tomato", "bowl"]
        chopped = [False]
        ori_pos = [[0, 5], [6, 5]]
        sentences = ["There is a fixed cutting board in the room."]
        in_plate = [False, False, False]

        item = []
        item_index = []
        plate_pos = obs[3:5]
        agent_pos = obs[9:11]
        first_cutting_board_pos = [1, 0]

        item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos}
        overlay = {"in_agent": [], "in_first_cutting_board": []}

        for i in range(2):
            pos = obs[3 * i: 3 * i + 2]
            if pos == ori_pos[i]:
                ingredient_in_ori_pos[i] == 1
                item.append(ingredient[i])
                item_index.append(i)

            if i < 1 and obs[3 * i + 2] == 3:
                chopped[i] = True

            for k in overlay.keys():
                if pos == item_pos[k]:
                    overlay[k].append(i)
        if len(item) == 1:
            template = "You notice {} on the table."
        elif len(item) == 2:
            template = "You notice {} and {} on the different tables."

        if len(item) > 0:
            sentences.append(template.format(*item).capitalize())

        cutting_board_index = ["first"]
        cutting_board_name = ["in_first_cutting_board"]

        cindex = 0
        if len(overlay[cutting_board_name[cindex]]) == 1:
            id = overlay[cutting_board_name[cindex]][0]
            template = "{} is on the cutting board."
            if id == 1:
                sentences.append(template.format("a bowl").capitalize())
            else:
                if chopped[id]:
                    sentences.append(template.format("a chopped " + raw_ingredient[id]).capitalize())
                else:
                    sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                if agent_pos == [cindex + 1, 1]:
                    action_list[-1] = "chop the " + raw_ingredient[id]


        elif len(overlay[cutting_board_name[cindex]]) > 1:

            full_plate_template = "a bowl containing a chopped tomato is on the cutting board."
            sentences.append(full_plate_template.capitalize())

            # in front of cutting board 1
        if agent_pos == [1, 1]:
            cindex = 0
        # in front of cutting board 2
        elif agent_pos == [2, 1]:
            cindex = 1
        else:
            cindex = -1

        action_template = "put the {} on the cutting board"
        hold_bowl_action = [
            "put the tomato in the bowl",
        ]

        if cindex >= 0:
            if len(overlay["in_agent"]) == 0:
                template = "Currently you are standing in front of the cutting board without anything in hand."
                sentences.append(template.format(cutting_board_index[cindex]).capitalize())

            elif len(overlay["in_agent"]) == 1:
                id = overlay["in_agent"][0]
                action_list[3] = "serve the dish"
                template = "Currently you are standing in front of the cutting board, carrying {} in hand."
                if id == 1:
                    sentences.append(template.format("a bowl").capitalize())
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format(raw_ingredient[id])
                else:
                    if chopped[id]:
                        sentences.append(template.format("a chopped " + raw_ingredient[id]).capitalize())
                    else:
                        sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                        action_list[2] = action_template.format(raw_ingredient[id])
            elif len(overlay["in_agent"]) > 1:
                action_list[3] = "serve the dish"
                in_plate_item = overlay["in_agent"][:-1]
                if len(in_plate_item) == 1:
                    full_plate_template = "Currently you are standing in front of the cutting board, carrying a bowl containing chopped {} in hand."
                sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                action_list[0] = hold_bowl_action[0]
                action_list[2] = action_template.format("bowl")
        else:
            if len(overlay["in_agent"]) == 0:
                template = "Currently you don't have anything in hand."
                sentences.append(template.format(cutting_board_index[cindex]).capitalize())

            elif len(overlay["in_agent"]) == 1:
                action_list[3] = "serve the dish"
                id = overlay["in_agent"][0]
                template = "Currently you are carrying {} in hand."
                if id == 1:
                    sentences.append(template.format("a bowl").capitalize())
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format(raw_ingredient[id])
                else:
                    if chopped[id]:
                        sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize())
                    else:
                        sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                        action_list[2] = action_template.format(raw_ingredient[id])
            elif len(overlay["in_agent"]) > 1:
                action_list[3] = "serve the dish"
                in_plate_item = overlay["in_agent"][:-1]
                if len(in_plate_item) == 1:
                    full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                elif len(in_plate_item) == 2:
                    full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                elif len(in_plate_item) == 3:
                    full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."
                sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                action_list[0] = hold_bowl_action[0]
                action_list[2] = action_template.format("bowl")

        sentences.append("To serve the dish of a bowl only containing chopped tomato, you should first")

    return {"prompt": " ".join(sentences), "action": action_list}
