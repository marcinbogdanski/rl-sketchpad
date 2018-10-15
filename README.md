# AI Sketchpad

Implementation of various AI algorithms

## Reinforcement Learning

### Reinforcement Learning: An Introduction (2nd ed, 2018) by Sutton and Barto

Implementation of selected algorithms from the book. Code snippets are as clean, minimal and close to the book as possible.

<!--* Chapter 1: Introduction -->
<!--  * Section 1.5: [Tic-Tac-Toe]() -->
Part I: Tabular Solution Methods
* Chapter 2: Multi-armed Bandits
  * Section 2.4: [Simple Bandit](RL_An_Introduction_2018/0204_Simple_Bandit.ipynb) - fig. 2.1, 2.2
  * Section 2.6: [Tracking Bandit](RL_An_Introduction_2018/0206_Tracking_Bandit.ipynb) - fig. 2.3
<!--  * Section 2.7: [UCB Bandit]() - plot figure 2.4 -->
<!--  * Section 2.8: [Gradient Bandit]() - plot figure 2.5 -->
<!--  * Section 2.10: [Bandit Parameter Study]() - plot figure 2.6 -->
* Chapter 4: Dynamic Programming
  * Section 4.1: [Iterative Policy Evaluation](RL_An_Introduction_2018/0401_Iterative_Policy_Evaluation.ipynb) - FrozenLake-v0 <!--gw 4.1 -->
  * Section 4.3: [Policy Iteration](RL_An_Introduction_2018/0403_Policy_Iteration.ipynb) - FrozenLake-v0 <!--gw, car-re, e4.2 f4.2 -->
  * Section 4.4: [Value Iteration](RL_An_Introduction_2018/0404_Value_Iteration.ipynb) - FrozenLake-v0 <!-- gambler problem, fig 4.3 -->
* Chapter 5: Monte Carlo Methods
  * Section 5.1: [First-Visit MC Prediction](RL_An_Introduction_2018/0501_First_Visit_MC_Prediction.ipynb) - Blackjack-v0, fig. 5.1
  * Section 5.3: [Monte Carlo ES Control](RL_An_Introduction_2018/0503_Monte_Carlo_ES_Control.ipynb) - Blackjack-v0, fig. 5.2
  * Section 5.4: [On-Policy First-Visit MC Control](RL_An_Introduction_2018/0504_On_Policy_First_Visit_MC_Control.ipynb) - Blackjack-v0
<!--  * Section 5.6: [Off-Policy MC Prediction](RL_An_Introduction_2018/0506_Off_Policy_MC_Prediction.ipynb) - fig. 5.3, 5.4-->
<!--  * Section 5.7: [Off-Policy MC Control](RL_An_Introduction_2018/0507_Off_Policy_MC_Control.ipynb) -->
<!--  * Section 5.8*: discounting aware IS -->
<!--  * Section 5.9*: per-decision IS -->
* Chapter 6: Temporal-Difference Learning
  * Section 6.1: [TD Prediction](RL_An_Introduction_2018/0601_TD_Prediction.ipynb) - Blackjack-v0, example 6.2, [Running-Mean MC Prediction](RL_An_Introduction_2018/0601_TD_Prediction.ipynb#Right-figure) alg.
  * Section 6.4: [Sarsa](RL_An_Introduction_2018/0604_Sarsa.ipynb) - WindyGridworld, example 6.5
  * Section 6.5: [Q-Learning](RL_An_Introduction_2018/0605_Q_Learning.ipynb) - CliffWalking, example 6.6
<!--  * Section 6.3: batch TD and MC - figure 6.2 -->
<!--  * Section 6.6: Expected Sarsa - figure 6.3, compare corridor? -->
<!--  * Section 6.7: Double Q-Learning - example 6.5 -->

<!--  * Section 7.1: N-Step TD Prediction - figure 7.2 -->
<!--  * Section 7.2: N-Step Sarsa - figure 7.4, +compare corridor? -->
<!--  * Section 7.3: Off-Policy N-Step Sarsa - importance sampling -->
<!--  * Section 7.5: N-Step Tree Backup -->
<!--  * Section 7.6: N-Step Q(phi?) - unifying algorithm -->

<!--  * Section 8.1: Random-Sample One-Step Tabular Q-Planning -->
<!--  * Section 8.2: Tabular Dyna-Q - dyna maze, fig 8.2, 8.3 -->
<!--  * Section 8.3: Fig 8.4, 8.5 (model is wrong) -->
<!--  * Section 8.4: Prioritized sweeping - example 8.4 -->
<!--  * Section 8.5: expected vs sample updates - figure 8.7 -->
<!--  * Section 8.6: trajectorry sampling - figure 8.8 -->
<!--  * Section 8.8, 8.9, 8.10, 8.11: planning at decision time - .. - MCTS -->

Part II: Approximate Solution Methods

* Chapter 9: On-Policy Prediction with Approximation
  * Section 9.3a: [Gradient Monte Carlo](RL_An_Introduction_2018/0903a_Gradient_MC.ipynb) - example 9.1, fig. 9.1 <!-- calc "true" -->
  * Section 9.3b: [Semi-Gradient TD](RL_An_Introduction_2018/0903b_Semi_Gradient_TD.ipynb) - example 9.2, fig. 9.2 (left)
  * Section 9.5a: [Linear Models - Polynomial and Fourier Bases](RL_An_Introduction_2018/0905a_LM_Poly_Fourier) - fig. 9.5

<!-- * Section 9.4 [N-Step Semi-Gradient TD]() - fig 9.2 (right)? -->

<!-- * Section 9.5 [Coarse Coding]() - square wave, fig 9.8 -->
<!-- * Section 9.5 [Tile Coding]() - fig 9.10 -->


* Linear Functions
  * Buckets
  * Tiles
* On-policy control - mountain car
* average reward [...]
* eligibility traces[...]
* policy gradients [...]


[//]: # (4.2, figure 4.1 - gridworld environment)
[//]: # (4.3, figure 4.2 - car rental env)
[//]: # (4.4, figure 4.3 - coin flip environment)

### UCL Course on RL (2016) Youtube lectures by David Silver

A bit more in-depth explanation of selected concepts from David Sivler lectures and Sutton and Barto book.

* Lecture 3 - Dynamic Programming
  * [Dynamic Programming](UCL_Course_on_RL/Lecture03_DP/DynamicProgramming.ipynb) - Iterative Policy Evaluation, Policy Iteration, Value Iteration
* Lecture 4 - Model Free Prediction
  * [MC and TD Prediction](UCL_Course_on_RL/Lecture04_Pred/ModelFreePrediction_Part1.ipynb)
  * [N-Step and TD(λ) Prediction](UCL_Course_on_RL/Lecture04_Pred/ModelFreePrediction_Part2.ipynb) - Forward TD(λ) and Backward TD(λ) with Eligibility Traces
* Lecture 4 - Model-Free Control
  * [On-Policy Control](UCL_Course_on_RL/Lecture05_Ctrl/ModelFreeControl_Part1.ipynb) - MC, TD, N-Step, Forward TD(λ), Backward TD(λ) with Eligibility Traces
  * [Off-Policy Control - Expectation Based](UCL_Course_on_RL/Lecture05_Ctrl/ModelFreeControl_Part2.ipynb) - Q-Learning, Expected SARSA, Tree Backup
  * [Off-Policy Control - Importance Sampling](UCL_Course_on_RL/Lecture05_Ctrl/ModelFreeControl_Part3.ipynb) - Importance Sampling SARSA, N-Step Importance Sampling SARSA, Off-Policy MC Control

