# AI Sketchpad

Implementation of various AI algorithms

## Reinforcement Learning

### Reinforcement Learning: An Introduction (2nd ed, 2018) by Sutton and Barto

Implementation of selected algorithms from the book. I tried to make code minimalistic and as close to the book as possible.

<!--* Chapter 1: Introduction -->
<!--  * Section 1.5: [Tic-Tac-Toe]() -->
Part I: Tabular Solution Methods
* Chapter 2: Multi-armed Bandits
  * Section 2.4: [Simple Bandit](RL_An_Introduction_2018/0204_Simple_Bandit.html) - plot figures 2.1, 2.2
  * Section 2.6: [Tracking Bandit](RL_An_Introduction_2018/0206_Tracking_Bandit.html) - plot figure 2.3
<!--  * Section 2.7: [UCB Bandit]() - plot figure 2.4 -->
<!--  * Section 2.8: [Gradient Bandit]() - plot figure 2.5 -->
<!--  * Section 2.10: [Bandit Parameter Study]() - plot figure 2.6 -->
* Chapter 4: Dynamic Programming
  * Section 4.1: [Iterative Policy Evaluation](RL_An_Introduction_2018/0401_Iterative_Policy_Evaluation.html) - uses FrozenLake-v0 <!-- add gridworld, figure 4.1 -->
  * Section 4.3: [Policy Iteration](RL_An_Introduction_2018/0403_Policy_Iteration.html) - uses FrozenLake-v0 <!-- add gridworld, example 4.2 Jack's car rental, figure 4.2 -->
  * Section 4.4: [Value Iteration](RL_An_Introduction_2018/0404_Value_Iteration.html) - uses FrozenLake-v0 <!-- gambler problem, figure 4.3 -->
* Chapter 5: Monte Carlo Methods
  * Section 5.1: [First-Visit MC Prediction](RL_An_Introduction_2018/0501_First_Visit_MC_Prediction.html) - figure 5.1
  * Section 5.3: [Monte Carlo ES Control](RL_An_Introduction_2018/0503_Monte_Carlo_ES_Control.html) - figure 5.2
  * Section 5.4: [On-Policy First-Visit MC Control](RL_An_Introduction_2018/0504_On_Policy_First_Visit_MC_Control.html)
<!--  * Section 5.6: __TODO__ [Off-Policy MC Prediction](RL_An_Introduction_2018/0506_Off_Policy_MC_Prediction.html)  figure 5.3 and 5.4 -->
<!--  * Section 5.7: __TODO__ [Off-Policy MC Control](RL_An_Introduction_2018/0507_Off_Policy_MC_Control.html) -->
<!--  * Section 5.8*: discounting aware IS -->
<!--  * Section 5.9*: per-decision IS -->
* Chapter 6: Temporal-Difference Learning
  * Section 6.1: [TD Prediction](RL_An_Introduction_2018/0601_TD_Prediction.html) - example 6.2, Blackjack-v0, includes [Running-Mean MC Prediction](RL_An_Introduction_2018/0601_TD_Prediction.html#Right-figure)
<!--  * Section 6.3: batch TD and MC - figure 6.2 -->
  * Section 6.4: [Sarsa](RL_An_Introduction_2018/0604_Sarsa.html)


[//]: # (4.2, figure 4.1 - gridworld environment)
[//]: # (4.3, figure 4.2 - car rental env)
[//]: # (4.4, figure 4.3 - coin flip environment)

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

