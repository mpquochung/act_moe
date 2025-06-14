
# Task: Sim transfer cube human
# python imitate_episodes.py \
#     --task_name sim_transfer_cube_human \
#     --ckpt_dir checkpoints/sim_transfer_cube_human/vanilla \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 5e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe False \

# python imitate_episodes.py \
#     --task_name sim_transfer_cube_human \
#     --ckpt_dir checkpoints/sim_transfer_cube_human/vanilla \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 5e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe False \
#     --eval

# python imitate_episodes.py \
#     --task_name sim_transfer_cube_human \
#     --ckpt_dir checkpoints/sim_transfer_cube_human/moe \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 3e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe True \

# python imitate_episodes.py \
#     --task_name sim_transfer_cube_human \
#     --ckpt_dir checkpoints/sim_transfer_cube_human/moe \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 3e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe True \
#     --eval


# Task: Sim transfer cube script

# python imitate_episodes.py \
#     --task_name sim_transfer_cube_scripted \
#     --ckpt_dir checkpoints/sim_transfer_cube_scripted/vanilla \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 5e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe False \
    
# python imitate_episodes.py \
#     --task_name sim_transfer_cube_scripted \
#     --ckpt_dir checkpoints/sim_transfer_cube_scripted/vanilla \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 5e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe False \
#     --eval

python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir checkpoints/sim_transfer_cube_scripted/moe \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 28 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 3e-5 \
    --seed 1000 \
    --num_experts 4 \
    --top_k 2 \
    --is_moe True \

python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir checkpoints/sim_transfer_cube_scripted/moe \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 28 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 3e-5 \
    --seed 1000 \
    --num_experts 4 \
    --top_k 2 \
    --is_moe True \
    --eval


# Task: Sim-insertion human

# 1. sim_insertion_human (train)
# python imitate_episodes.py \
#     --task_name sim_insertion_human \
#     --ckpt_dir checkpoints/sim_insertion_human/vanilla \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 5e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe False

# python imitate_episodes.py \
#     --task_name sim_insertion_human \
#     --ckpt_dir checkpoints/sim_insertion_human/vanilla \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 5e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe False\
#     --eval

# 2. sim_insertion_scripted (train, vanilla)
# python imitate_episodes.py \
#     --task_name sim_insertion_scripted \
#     --ckpt_dir checkpoints/sim_insertion_scripted/vanilla \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 5e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe False

# 3. sim_insertion_scripted (eval, vanilla)
# python imitate_episodes.py \
#     --task_name sim_insertion_scripted \
#     --ckpt_dir checkpoints/sim_insertion_scripted/vanilla \
#     --policy_class ACT \
#     --kl_weight 10 \
#     --chunk_size 100 \
#     --hidden_dim 512 \
#     --batch_size 64 \
#     --dim_feedforward 3200 \
#     --num_epochs 2000 \
#     --lr 5e-5 \
#     --seed 1000 \
#     --num_experts 4 \
#     --top_k 2 \
#     --is_moe False \
#     --eval

# 4. sim_insertion_scripted (train, moe)
python imitate_episodes.py \
    --task_name sim_insertion_scripted \
    --ckpt_dir checkpoints/sim_insertion_scripted/moe \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 28 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 3e-5 \
    --seed 1000 \
    --num_experts 4 \
    --top_k 2 \
    --is_moe True

# 5. sim_insertion_scripted (eval, moe)
python imitate_episodes.py \
    --task_name sim_insertion_scripted \
    --ckpt_dir checkpoints/sim_insertion_scripted/moe \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 28 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 3e-5 \
    --seed 1000 \
    --num_experts 4 \
    --top_k 2 \
    --is_moe True \
    --eval

# 1. sim_insertion_human (train, MoE)
python imitate_episodes.py \
    --task_name sim_insertion_human \
    --ckpt_dir checkpoints/sim_insertion_human/moe \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 28 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 3e-5 \
    --seed 1000 \
    --num_experts 4 \
    --top_k 2 \
    --is_moe True

# 2. sim_insertion_human (eval, MoE)
python imitate_episodes.py \
    --task_name sim_insertion_human \
    --ckpt_dir checkpoints/sim_insertion_human/moe \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 28 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 3e-5 \
    --seed 1000 \
    --num_experts 4 \
    --top_k 2 \
    --is_moe True \
    --eval