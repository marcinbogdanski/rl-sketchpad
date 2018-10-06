# AI Sketchpad

Implementation of various AI algorithms

## Reinforcement Learning

### Reinforcement Learning: An Introduction (2nd ed, 2018) by Sutton and Barto

Implementation of selected algorithms from the book. I tried to make code minimalistic and as close to the book as possible.

Part I: Tabular Solution Methods
* Chapter 2: Multi-armed Bandits
  * Section 2.4: [Simple Bandit](RL_An_Introduction_2018/0204_Simple_Bandit.html) - figures 2.1, 2.2
  * Section 2.6: [Tracking Bandit](RL_An_Introduction_2018/0206_Tracking_Bandit.html) - figure 2.3
* Chapter 4: Dynamic Programming
  * Section 4.1: [Iterative Policy Evaluation](RL_An_Introduction_2018/0401_Iterative_Policy_Evaluation.html) - uses FrozenLake-v0
  * Section 4.3: [Policy Iteration](RL_An_Introduction_2018/0403_Policy_Iteration.html) - uses FrozenLake-v0
  * Section 4.4: [Value Iteration](RL_An_Introduction_2018/0404_Value_Iteration.html) - uses FrozenLake-v0


### UCL Course on RL
By David Silver
* Lecture 3 - Dynamic Programming
  * [Dynamic Programming](UCL_Course_on_RL/Lecture03_DP/DynamicProgramming.html) - Iterative Policy Evaluation, Policy Iteration, Value Iteration
* Lecture 4 - Model Free Prediction
  * [MC and TD Prediction](UCL_Course_on_RL/Lecture04_Pred/ModelFreePrediction_Part1.html)
  * [N-Step and TD(λ) Prediction](UCL_Course_on_RL/Lecture04_Pred/ModelFreePrediction_Part2.html) - Forward TD(λ) and Backward TD(λ) with Eligibility Traces
* Lecture 4 - Model-Free Control
  * [On-Policy Control](UCL_Course_on_RL/Lecture05_Ctrl/ModelFreeControl_Part1.html) - MC, TD, N-Step, Forward TD(λ), Backward TD(λ) with Eligibility Traces
  * [Off-Policy Control - Expectation Based](UCL_Course_on_RL/Lecture05_Ctrl/ModelFreeControl_Part2.html) - Q-Learning, Expected SARSA, Tree Backup
  * [Off-Policy Control - Importance Sampling](UCL_Course_on_RL/Lecture05_Ctrl/ModelFreeControl_Part3.html) - Importance Sampling SARSA, N-Step Importance Sampling SARSA, Off-Policy MC Control

