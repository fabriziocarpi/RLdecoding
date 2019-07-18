# ============================================================== #
# imports {{{
# ============================================================== #
import gym
import sys
from gym import spaces
import numpy as np
import scipy.io as sio
import json # json.loads
import os # os.listdir
import re # re.findall
import ray
import ray.rllib.agents.dqn as dqn
import argparse as ap

from CodeEnv import *
# }}}
# ============================================================== #
# parameters {{{
# ============================================================== #
tmpdir = "/tmp/fc94"


parser = ap.ArgumentParser("python eval.py")
parser.add_argument("dB_range", help="set of SNRs, e.g., [5] or [2,3,4]")
parser.add_argument("minCwErr", help="get at least this many codeword errors")
parser.add_argument("maxCw", help="but stop if this many codewords have been simulated")
parser.add_argument("NUM", help="number appended to the name of the result file")
parser.add_argument("--path", dest="path_to_results", default="./latest", help="path to the result directory")
parser.add_argument('--save', dest='SAVE', action='store_true')
parser.add_argument('--no-save', dest='SAVE', action='store_false')
parser.set_defaults(SAVE=True)

args = parser.parse_args()

dB_range = np.asarray(eval(args.dB_range))
minCwErr = int(args.minCwErr)
maxCw = int(args.maxCw)
NUM = int(args.NUM)
path_to_results = args.path_to_results
SAVE = args.SAVE

if SAVE:
    save_path =  path_to_results+"/res_{}.mat".format(NUM)
    save_path_txt =  path_to_results+"/res_{}.txt".format(NUM)

# }}}
# ============================================================== #
with open(path_to_results + "/params.json") as h:
    config = json.loads(h.read())

env_config = config["env_config"]

# find all checkpoint and load the latest
filenames = os.listdir(path_to_results)
checkpoint_numbers = []
for fn in filenames:
    m = re.findall('checkpoint_(\d+)', fn)
    if not m: continue
    print(m[0])
    checkpoint_numbers.append(int(m[0]))

mc = max(checkpoint_numbers)
checkpoint_path = path_to_results+"/"+"checkpoint_{}/checkpoint-{}".format(mc,mc)
print("found {} checkpoints".format(len(checkpoint_numbers)))
print("restoring "+checkpoint_path)

# ============================================================== #
# evaluation {{{
# ============================================================== #
#ray.init()
ray.init(temp_dir=tmpdir+"/ray")  # you may need to change the temp directory in case it runs on a cluster or shared machine

if config["optimizer_class"] == "AsyncReplayOptimizer":
    trainer = dqn.ApexTrainer(config=config, env=CodeEnv)
else:
    trainer = dqn.DQNTrainer(config=config, env=CodeEnv)
trainer.restore(checkpoint_path)
env = CodeEnv(env_config)
n = env.n

dB_len = len(dB_range)
BitErr = np.zeros([dB_len], dtype=int)
CwErr = np.zeros([dB_len], dtype=int)
totCw = np.zeros([dB_len], dtype=int)
totBit = np.zeros([dB_len], dtype=int)

for i in range(dB_len):
    print("\n--------\nSimulating EbNo = {} dB".format(dB_range[i]))
    env.set_EbNo_dB(dB_range[i])

    while(CwErr[i]<minCwErr and totCw[i]+1<=maxCw):
        obs = env.reset()
        done = (env.syndrome.sum() == 0)

        while not done:
            action = trainer.compute_action(obs)
            obs, _, done, _ = env.step(action)
            #env.render()

        BitErrThis = np.sum(env.chat)
        BitErr[i] = BitErr[i] + BitErrThis
        if BitErrThis > 0:
            CwErr[i] = CwErr[i] + 1

        totCw[i] += 1
        totBit[i] += n

    print("CwErr:", CwErr[i])
    print("BitErr:", BitErr[i])
    print("TotCw:", totCw[i])
    print("CER:", CwErr[i]/totCw[i])
    print("BER:", BitErr[i]/totBit[i])

if SAVE:
    resdict = {
        "dB_range": dB_range,
        "CwErr": CwErr,
        "BitErr": BitErr,
        "TotCw": totCw,
        "TotBit": totBit,
    }

    print("\n****\nSaving files to:\n.mat -->"+save_path+"\n.txt -->"+save_path_txt)
    sio.savemat(save_path, resdict)
    with open(save_path_txt, 'w') as file_txt:
        file_txt.write(str(resdict))

ray.shutdown()

print("done!")

# }}}
# ============================================================== #
