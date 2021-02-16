# Reinforcement learning implementations and experiments

This repository contains implementations of reinforcement learning algorithms
and experiments with environments using the OpenAI Gym API.


## Algorithms

### Dynamic programming
* [Policy evaluation](algorithms/dp_policy_evaluation.py)
* [Policy iteration](algorithms/dp_policy_iteration.py)
* [Value iteration](algorithms/dp_value_iteration.py)

### Monte Carlo
* [Monte Carlo on-policy evaluation](algorithms/mc_on_policy_evaluation.py)
* [Monte Carlo on-policy control](algorithms/mc_on_policy_control.py)
* [Monte Carlo off-policy evaluation](algorithms/mc_off_policy_evaluation.py)
* [Monte Carlo off-policy control](algorithms/mc_off_policy_control.py)


### Temporal-difference
* [TD(0)](algorithms/td0.py)
* [SARSA](algorithms/sarsa.py)
* [Expected SARSA](algorithms/expected_sarsa.py)
* [Q-Learning](algorithms/qlearning.py)
* [Double Q-Learning](algorithms/double_qlearning.py)

## Environments
* [2d grid world](environments/gridworld/): A 2d rectangular grid world where agent can move deterministically up, down, left, right.


## Experiments
* [N-armed Bandit](experiments/bandit)
* [Gridworld](experiments/gridworld)
* [Frozen Lake](experiments/frozenlake)
* [Taxi](experiments/taxi)