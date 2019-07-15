import gym
import numpy as np
import scipy.io as sio
import os
from pathlib import Path
from utils import *

# default for maxIter : env.spec.max_episode_steps

class CodeEnv(gym.Env):
    def __init__(self, env_config): # RLLIB: custom env classes must take a single config parameter
        """
        Args:
            code: string that identifies the parity-check matrix, e.g., "RM_2_5_std"
            maxIter: maximum number of decoding iterations
            EbNo_dB: training signal-to-noise ratio (SNR) in dB

            (optional):
            WBF: weighted bit-flipping (default: False)
            Lmax: maximum LLR value (default: 2.0)
            asort: automorphism sort (default: False), strategy depends on the code (either RM or BCH)
            asortk: if asort==True for BCH codes (default: -2)
            path_to_Hmat: path to .mat file for H (default: "~/Hmat")
            Hrank: rank of the parity-check matrix (default: will be computed which may take some time)
        """

        self.code = env_config["code"]
        #path_to_Hmat = env_config.get("path_to_Hmat", "default")
        if "path_to_Hmat" in env_config:
            path_to_Hmat = Path(env_config["path_to_Hmat"])
        else:
            path_to_Hmat = Path.home() / "Hmat"
            print("CodeEnv: path_to_Hmat not specified. Looking in "+str(path_to_Hmat))

        # optional parameters
        self.WBF = env_config.get("WBF", False)
        self.Lmax = env_config.get("Lmax", 2.0)
        self.asort = env_config.get("asort", False)

        # environment setup
        H = sio.loadmat(str(path_to_Hmat / env_config["code"]))['H']
        H = np.int64(H)
        self.H = H
        self.m = self.H.shape[0] # number of parity checks
        self.n = self.H.shape[1] # code length 
        print("determining rank(H) ...")
        self.Hrank = env_config.get("Hrank", gfrank(self.H)) # n-k
        print("rank(H) = {}".format(self.Hrank))
        self.k = self.n - self.Hrank # code dimension
        self.R = self.k/self.n # code rate
        self.action_space = gym.spaces.Discrete(self.n)
        self.absL_avg = np.zeros(shape=self.n) # running average of abs(LLRs)
        self.path_penality = np.zeros(shape=self.n)
        self.totCw = 0 # number of total codewords = number of resets

        self.maxIter = env_config["maxIter"]
        self.set_EbNo_dB(env_config["EbNo_dB"])

        if self.WBF == True:
            # (s,r)
            #P1 = gym.spaces.Tuple((gym.spaces.Discrete(2),)*self.m) # binary tuple/vector
            #P2 = gym.spaces.Box(low=0, high=self.Lmax, shape=[self.m], dtype=np.float32) # binary tuple/vector
            #self.observation_space = gym.spaces.tuple((P1, P2))

            self.observation_space = gym.spaces.Box(low=-self.Lmax, high=self.Lmax, shape=[self.m], dtype=np.float32) # binary tuple/vector
            self.CN_neighbors = [self.H[i,:].nonzero() for i in range(self.m)] 
        else:
            self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(2),)*self.m) # binary tuple/vector

        self.reset()

    def set_EbNo_dB(self, x):
        self.EbNo_dB = x
        self.sigma2 = 1/(2*self.R*10**(self.EbNo_dB/10))

    def reset(self): # all-zero codeword transmission
        c = np.zeros(self.n, dtype=np.int)
        y = (-2*c+1) + np.random.normal(scale=np.sqrt(self.sigma2), size=self.n)

        if self.asort == True:
            if "RM" in self.code:
                sind = find_rm_auto(y)
            elif "BCH" in self.code:
                sind = find_cyclic_auto(y,-2)
            else:
                raise ValueError("code string must contain either RM or BCH if asort==True")
            y = y[sind]

        l = 2*y/self.sigma2 # LLRs
        z = np.zeros(self.n, dtype=np.int)
        z[y<0] = 1 # hard decision observation

        # if asort == False, this should converge to uniform
        self.absL_avg = (abs(l) + self.totCw*self.absL_avg)/(self.totCw + 1)

        self.chat = z # current codeword estimate (including bit flips)
        self.error_locations = np.where(z==1)[0]
        self.nerr = len(self.error_locations)

        # generate syndrome
        self.syndrome = np.zeros(self.m, dtype=np.int64)
        for i in self.error_locations:
            self.syndrome = (self.syndrome + self.H[:,i]) % 2
        
        if self.WBF == True: # compute soft syndrome
            self.r = np.abs(y)
            self.phi = [self.r[self.CN_neighbors[i]].min() for i in range(self.m)]
            self.soft_syndrome = (-2*self.syndrome+1) * self.phi
            self._state = self.soft_syndrome
            self.path_penality = - abs(l) / np.mean(self.absL_avg) * 1.0/self.maxIter

            # new
            #self._state = tuple(tuple(self.syndrome), self.r)
        else:
            self._state = tuple(self.syndrome)
            self.path_penality = - self.absL_avg / np.mean(self.absL_avg) * 1.0/self.maxIter

        self.nmove = 0
        self.totCw += 1
        #return self._get_obs()
        return self._state

#    def _get_obs(self):
#        if self.WBF == True:
#            return self._state # for Box, np.arrays are required as input
#        else:
#            return tuple(self._state)

    def step(self, action): # see http://gym.openai.com/docs/#observations
        self.nmove += 1
        self.chat[action] = (self.chat[action] + 1) % 2
        self.syndrome = (self.syndrome + self.H[:, action]) % 2
        if self.WBF == True:
            # update reliabilities + recompute phi 
            #self.r[action] = self.Lmax
            #self.phi = [self.r[self.CN_neighbors[i]].min() for i in range(self.m)]

            self._state = (-2*self.syndrome+1) * self.ph

            # new
            #self._state = tuple(tuple(self.syndrome), self.r)
        else:
            self._state = tuple(self.syndrome)

        if (self.syndrome.sum() == 0):
            done = True
            reward = self.path_penality[action] + 1
        else:
            done = False
            reward = self.path_penality[action]

        #if self.WBF == True: # update path penalities
        #    self.path_penality[action] = -self.Lmax # may be too high

        if self.nmove >= self.maxIter:
            done = True

        return self._state, reward, done, {}

    def render(self):
        print('move:',self.nmove,'chat:',self.chat,' state:', self._state)
        #print('nerr:',self.nerr,' err loc',self.errloc,' nmove:',self.nmove,' state:', self._state)
