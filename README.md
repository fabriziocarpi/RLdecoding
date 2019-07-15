# CodeEnv: Learning Bit-Flipping Decoding

This repository includes a collection of Python and Tensorflow scripts to learn Bit Flipping decoding through Reinforcement Learning.

Additional details are explained in this paper:

* F. Carpi, C. Häger, M. Martalò, R. Raheli, H. D. Pfister, "[Reinforcement
  Learning for Channel Coding: Learned Bit-Flipping
  Decoding](http://arxiv.org/pdf/tbd.pdf)", arxiv, June 2019

---

## Parity Check matrices

The Parity Check (PC) matrices for the following codes are contained in the `Hmat` folder:
 - Reed–Muller (RM) codes:
   - RM(2,5)
   - RM(3,6)
   - RM(3,7)
   - RM(4,7)

 - Bose–Chaudhuri–Hocquenghem (BCH) codes:
   - BCH(63,45)

The suffix denotes:
 - `_std`: standard PC matrix
 - `_oc`: overcomplete PC matrix (the rows are dual codewords)

## Getting Started

The main code is contained in the files:

* `CodeEnv.py`: environment class based on [OpenAI
  Gym](https://gym.openai.com/)
* `utils.py`: various helper functions (mainly to implement the permutation
  preprocessing based on the code's automorphism group)

The actual optimization uses
[RLlib](https://ray.readthedocs.io/en/latest/rllib.html) and
[Tune](https://ray.readthedocs.io/en/latest/tune.html), both of which are based
on [Ray](https://ray.readthedocs.io/en/latest/index.html).

## Workflow

### 0) Initialization
Make sure Ray and RLlib are installed. An example of Anaconda environment is provided in `rllib.yml`.


### 1) Update config settings and log folder

Copy the folder `Hmat` to a desired directory `<MY_Hmat_PATH>`.
In `opt.py`, update the config dictionary settings and set the log folder in the call to `tune.run()`.

> "path_to_Hmat" : "<MY_Hmat_PATH>",   
> ray.init(temp_dir='<TEMP_RAY_DIR>'),    
> local_dir="<RAY_RESULTS_DIR>",    

Note that, in `opt.py`, accordingly to computation capabilities, the number of CPU/GPU may be edited as well:
> "num_gpus" : <N_GPUs>,    
> "num_workers": <N_CPUs>



### 2) Run the optimization and explore hyperparameters

Use `optscript` to run `opt.py` from the terminal. This file just writes the process ID to the file `last_pid` and calls `opt.py`. In case something goes wrong, use `kill_last_pid` to terminate this process and all child processes created by Ray.

Note that a new directory `<TRAINED_AGENT_DIR>`, containing all the model's files, will be created:
> <RAY_RESULTS_DIR>/CodeEnv/<TRAINED_AGENT_DIR>


Tensorboard can be used with the specified log directory to monitor the training progress.


### 3) Evaluate the trained model

Finally, use `jobscript` to evaluate the trained model for a specified range of SNRs. This script calls `eval.py` which then saves the results into .mat files.

Modify `jobscript`, inserting the proper folder for your trained model:
> resultPath="<RAY_RESULTS_DIR>/CodeEnv/<TRAINED_AGENT_DIR>"    


Change in `eval.py`:
> ray.init(temp_dir='<TEMP_RAY_DIR>')    


---
### Notes

The current scripts replicate the results for RM (2,5) [LBF-NN] and [s+d LBF-NN].

Please send any feedback, comment or request to [Fabrizio Carpi](https://fabriziocarpi.github.io/) or [Christian Häger](http://www.christianhaeger.de/index.html).
