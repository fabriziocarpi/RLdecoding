# ============================================================== #
# imports {{{
# ============================================================== #
import sys
import ray
import ray.rllib.agents.dqn as dqn
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.trainer import with_common_config
from ray import tune
from CodeEnv import *
# }}}
# ============================================================== #
# config {{{
# ============================================================== #

config = with_common_config(
    {
    "env_config": {
        "code" : "RM_2_5_std",     # Select code
        #"code" : "BCH_63_45_oc",   # Example of other code
        "EbNo_dB" : 4,   # training SNR
        "maxIter" : 10,  # maximum iterations (agent's life)
        "WBF" : False,   # weighted bit-flipping
        "asort" : True, # automorphism sort
		"path_to_Hmat" : "/home/fc94/RL_source/Hmat"
    },
    "env": CodeEnv,
    "log_level": "WARN",

    # === Model ===
    # Number of atoms for representing the distribution of return. When
    # this is greater than 1, distributional Q-learning is used.
    # the discrete supports are bounded by v_min and v_max
    "num_atoms": 1,
    "v_min": -10.0,
    "v_max": 10.0,
    "noisy": False, # Whether to use noisy network
    "sigma0": 0.5,  # control the initial value of noisy nets
    "dueling": False,   # Use dueling network
    "double_q": False,  # Use double Q-learning 
    # Postprocess model outputs with these hidden layers to compute the
    # state and action values. See also the model config in catalog.py.
    #"hiddens": [256],
    "n_step": 1, # N-step Q learning
    "model": {
        "fcnet_activation": "relu",
        #"fcnet_activation": tune.grid_search(["tanh", "relu"]),
        "fcnet_hiddens": [500],
        #"fcnet_hiddens": tune.grid_search([[1500],[1000],[500]]),
#        "l2_reg": 1e-6,
    },

    # see https://github.com/ray-project/ray/blob/cff08e19ff1606ef6e718624703e8e0da19b223d/python/ray/rllib/agents/dqn/dqn.py#L257-L261 
    "timesteps_per_iteration": 1000, # Number of env steps to optimize for before returning
                                     # one optimiziation step uses batch.count env steps for DQN
    "min_iter_time_s": 10, # prevent iterations from going lower than this time span

    # === Exploration ===
    # Max num timesteps for annealing schedules. Exploration is annealed from
    # 1.0 to exploration_final_eps over this number of timesteps multiplied by
    # exploration_fraction
    "schedule_max_timesteps": 2500000,
    # Fraction of entire training period over which the exploration rate is
    # annealed
    "exploration_fraction": 0.9,
    # Final value of random action probability
    "exploration_final_eps": 0.0,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 50000,
    # Use softmax for sampling actions. Required for off policy estimation.
    #"soft_q": tune.grid_search([True, False]),
    "soft_q": False,
    # Softmax temperature. Q values are divided by this value prior to softmax.
    # Softmax approaches argmax as the temperature drops to zero.
    "softmax_temp": 1.0,
    # If True parameter space noise will be used for exploration
    # See https://blog.openai.com/better-exploration-with-parameter-noise/
    "parameter_noise": False,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    #"buffer_size": 200000,
    "buffer_size": 10000,
    # If True prioritized replay buffer will be used.
    #"prioritized_replay": tune.grid_search([True, False]),
    "prioritized_replay": False,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Fraction of entire training period over which the beta parameter is
    # annealed
    "beta_annealing_fraction": 0.2,
    # Final value of beta
    "final_prioritized_replay_beta": 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,
    # Whether to LZ4 compress observations
    "compress_observations": True,

    # === Optimization ===
    # Learning rate for adam optimizer
    "lr": 0.0001,
    #"lr": tune.grid_search([0.0001, 0.00007]),
    # Learning rate schedule
    "lr_schedule": None,
    # Adam epsilon hyper parameter
    "adam_epsilon": 1e-8,
    "grad_norm_clipping": 40, # If not None, clip gradients during optimization at this value
    "learning_starts": 1000, # How many steps of the model to sample before learning starts.
    # Update the replay buffer with this many samples at once. Note that
    # this setting applies per-worker if num_workers > 1.
    "sample_batch_size": 50, # APEX
    #"sample_batch_size": 4,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 32,
    #"train_batch_size": tune.grid_search([32, 64]),

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_gpus": 1,
    "num_workers": 10,
    #"num_gpus": 1, # APEX
    #"num_workers": 10, # APEX
#    "optimizer_class": "SyncReplayOptimizer", # Optimizer class to use.
    # APEX
    "optimizer_class": "AsyncReplayOptimizer", # for APEX / does not work with DQN (only APEX)
    "optimizer": {
            "max_weight_sync_delay": 2000,
            "num_replay_buffer_shards": 5, # increasing this increases GPU usage
            "debug": False
        },
    "per_worker_exploration": False, # Whether to use a distribution of epsilons across workers for exploration.
    #"per_worker_exploration": True, # APEX
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    #"worker_side_prioritization": True, #APEX
    },
)

# }}}
# ============================================================== #
# run optimizations {{{
# ============================================================== #
#ray.init()
ray.init(temp_dir='/tmp/fc94/ray')  # you may need to change the temp directory in case it runs on a cluster or shared machine


tune.run(
    "APEX",
    #"DQN",
    name="CodeEnv",
    checkpoint_at_end=True,
    num_samples=1,
    local_dir="/home/fc94/RL_source/ray_results",
    #stop={"episode_reward_mean": 10},
    stop={"training_iteration": 150},
    config=config
)

ray.shutdown()

print("done!")
# }}}
# ============================================================== #
