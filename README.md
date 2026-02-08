# Rover Simulation in a 20x20 Grid World

This project simulates a rover in a 20x20 grid world with blocked cells. The rover has a stochastic transition model.

## Project Goal

The goal of this project is to implement and compare two reinforcement learning agents for path planning in this environment:

1.  A Passive ADP (Adaptive Dynamic Programming) Agent with known transition probabilities.
2.  An Active Q-Learning Agent with epsilon-greedy exploration.

## Key Features

- **Grid World:** A 20x20 grid with randomly placed obstacles.
- **Stochastic Motion:** The rover's moves are not deterministic (0.8 probability of intended move, 0.1 for perpendicular slips).
- **Path Generation:** Generation of 10 unique source-destination pairs.
- **Agent Comparison:** Comparison of trajectories based on path length, convergence, and obstacle avoidance.
- **Transition Matrix Estimation:** Estimation of the transition probability matrix learned by the ADP agent and comparison with the true matrix.
- **Visualization:** Visualization of trajectories and transition-matrix differences.
