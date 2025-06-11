### Environments:

1. ALFWorld

Install LLF-Bench including ALFWorld by https://github.com/microsoft/LLF-Bench. The number of the alfworld scenarios can be tuned by the parameters in llfbench/envs/alfworld/baseconfig.yaml:

```
dataset:
  num_train_games: 50                                         # max training games (<=0 indicates full dataset)
  num_eval_games: 50     
  
env:
  task_types: [1] #[1, 2, 3, 4, 5, 6]                               # task-type ids: 1 - Pick & Place, 2 -Examine in Light, 3 - Clean & Place, 4 - Heat & Place, 5 - Cool & Place, 6 - Pick Two & Place

```

We also provide one group of config of our main results in 'envs/alfworld/baseconfig.yaml' for reference.

2. Overcooked

Install textual Overcooked (gym-macro-overcooked package) following https://github.com/WeihaoTan/TWOSOME



### Value-based LLM-Prior



1. DQN-Prior for ALFWorld

```
### train
python train_alfworld_thought.py --epoch 1000 --eval_interval 80 --target_update 20 --update_type hard --epsilon 0.2 --lr 5e-4 --batch 128 --capacity 20000 --train_iter 40 --device 0 --action_normalize True --print_interval 1 --cql_coef 0.0  --checkpoint_path alfworld50tasks
### test

SAVED_QNETWORK="alfworld50tasks/QProb/qprob_current_best.pt"
python test_alfworld.py --epoch 800 --eval_interval 40 --target_update 10 --lr 5e-4 --batch 64 --capacity 20000 --train_iter 1000 --device 1 --action_normalize True --print_interval 1 --cql_coef 1.0 --gamma 0.99 --checkpoint_path test_alfworld --reference_llm_path xxx --num_gens 15 --gen_batch_size 15 --temperature 0.8 --top_p 0.95 --saved_qnetwork "$SAVED_QNETWORK"

```

2. DQN-Prior for Overcooked

```
python train_overcooked.py --epoch 200 --eval_interval 20 --target_update 20 --update_type soft --epsilon 0.2 --lr 5e-4 --batch 64 --capacity 4000 --train_iter 40 --device 0 --action_normalize True --print_interval 1 --cql_coef 0.0  --seed 1234 --num_gens 5 --gen_batch_size 5 --alg dqn --alpha 10 --checkpoint_path overcooked

python test_overcooked_salad.py --epoch 300 --eval_interval 20 --target_update 20 --update_type soft --epsilon 0.2 --lr 5e-4 --batch 64 --capacity 4000 --train_iter 40 --device 0 --action_normalize True --print_interval 1 --cql_coef 0.0  --seed 1234 --num_gens 5 --gen_batch_size 5 --alg prior --checkpoint_path overcooked_salad

```



3. CQL-Prior for Overcooked(Salad)

```
python cql_overcooked_salad --epoch 800 --eval_interval 40 --target_update 20 --lr 5e-4 --batch 128 --capacity 20000 --train_iter 10000 --device 1 --action_normalize True --print_interval 1 --cql_coef 5.0 --gamma 0.99 --update_type soft --mse_type sum --off_type prior   --number 2000 --checkpoint_path  offlineovercooked

```

