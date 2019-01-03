# Pommerman
Multi-Agent Learning with Pommerman

Pommerman is a multi-agent environment based on the classic console game Bomberman (Nintendo 1983). Every battle starts on a randomly drawn symmetric 11x11 grid with four agents each initially located in one of the four corners. In every timestep the agents can take one of six different actions involving moving, bombing or stop/no action. Besides the agents, the board consist of both destructable(wooden) and indestructible(rigid) walls, when bombing a wooden wall it reveals a new passage, and there is a 50% change of dropping a power-up. There are three different power-ups in the game: ”Extra Ammo”, ”Extra Blast Range” and ”Can Kick”, referring to the agents bomb or ability to kick bombs. There are three ways of playing the game: free-for-all (FFA), Teams, and Teams with radio. This paper is going to focus only on the FFA game mode and a use of a single agent. Multiagent game mode assumes a working single agent - as a result the main focus is with a single agent. Throughout this paper, there will be investigated different actor models and network architectures for solving the problem of adding reinforcement learning to a single agent.

In this project we showcase different approaches to the Pommerman problem. We will investigate several methods and techniques which can be applied to train a single agent, that competes against three opposing agents in the ”free-for- all” mode of the game. The cooperation of multiple agents should not be the main focus of this research.

## Usage

- **demonstration.ipynb** Implementation of the core algorithms and methods we applied.

- **evaluation.ipynb** Plotting of some of our main results.

Please refer to Resnick et al. 2018 at https://www.pommerman.com/ and https://github.com/MultiAgentLearning/playground
