# .ipynb files

## 1.Implementation_SARSA_Qlearning_DQN.ipynb
This file includes all codes to implement Deep SARSA, Q-learning, and DQN models. You can implement all code before [1 SARSA], and then run all the code contained in the model you want.

## 2.Hyperparameter_modification.ipynb
This file is a template for the experiments on Hyperparameter modification. You can change the two hyperparameters(gamma and beta) and run this file to get corresponding results.
```
# HYPERPARAMETERS WHICH CAN BE CHANGED 
gamma = 0.95  # THE DISCOUNT FACTOR
beta = 0.005  # THE PARAMETER SETS HOW QUICKLY THE VALUE OF EPSILON IS DECAYING
```
Please see the description in this file for more details.

## 3.Reward_administration.ipynb
This file is a template for the experiments on Reward administration. You can change the hyperparameter r_draw(reward of draw) and run this file to get corresponding results.
```
## SET DIFFERENT REWARDS
r_checkmate = 1       ## reward of checkmate
r_draw = -0.9         ## reward of draw
```
Please see the description in this file for more details.

## 4.figures_hyperparameters&rewards.ipynb
This file contains all figures of Hyperparameter modification and Reward administration. It uses the experimental results from the folders "Results_diff_hyperparameters" and "Results_diff_rewards".

## 5.Implementation_best_SARSA_Qlearning_DQN.ipynb
This file includes all codes to implement the best Deep SARSA, Q-learning, and DQN models with the relatively best hyperparameters.
You can implement all code before [1 SARSA], and then run all the code contained in the model you want.

## 6.gradient_exploding.ipynb and gradient_exploding_clipping.ipynb
These two files are designed to solve gradient exploding problems via a comparison experiment. The second one use a global norm clipping method to solve this problem.

# .py files

## 1.Chess_env_reward_change.py
This file is a new environment for the agent. We refactored its constructor so that it can accept two more parameters (r_checkmate; r_draw). We used this new environment in Reward_administration.ipynb.
