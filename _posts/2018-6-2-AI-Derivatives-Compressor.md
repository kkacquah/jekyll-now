---
layout: post
mathjax: true
title: My First Post Derivatives Portfolio Compression With Deep Learning (Part One)

---

![Compression Example](https://raw.githubusercontent.com/kkacquah/kkacquah.github.io/master/images/AI-Derivatives-Compressor/compression.png)

For my UROP project for the spring 2018 semester, I have created an Open AI Gym environment in an attempt to open the door for open source the problem of using reinforcement learning to build an automatic agent that finds the optimal policy for compressing a set of derivatives contracts between groups of banks. Given the recent resurgence of financial regulation incentivizing actors within the financial sector to minimize their excess derivative contract notional, devising an artificial intellegence system to automatically minimize this excess over time is certainly a worthwhile endevour.

## Problem Statement

Perhaps I should define the problem I intend to solve, before providing information about the artificial intelligence mechanisms I used to achieve a solution. In this notebook, I attempt to build an OpenAI Gym (a standard format for testing and evaluating reinforcement agents' attempts at various problems) that can be used to test a reinforcement learning agent which aims to solve the following problem:

First, we must define the following notation, a derivatives market can be modelled as consistenting the following elements:

1. A set of banks $B$ comprised of banks $(b_0,b_1,\dots b_n)$
1. A set of timesteps $T = (0,1,2,3,4,5,\dots,n)$, each timestep $t$ represents a week during which derivatives trading occurs.
1. A set of edges that comprise a directed graph $G$, $E = (e_0,e_1, \dots e_n)$ each edge $e_k =(b_i,b_j)$ of weight $w$ represents an debt owed of $w$ million dollars from bank $b_i$ to bank $b_j$.
1. An "arrival" of an edge is defined as one of two events:
    1. The origination of an edge $e_k =(b_i,b_j)$ that did not exist in prior timesteps.
    1. The increase of the weight $w$ of an edge $e_k$ that had existed in prior timesteps.
1. The total notional of a derivatives contract graph $G$ is equivalent to:
$$\sum_{\forall i,j \in |B|} w(b_i,b_j)$$

Where $w(b_i,b_j)$ is a function that returns the weight $w$ of the edge defined as a derivative contract relationship between bank $i$ and bank $j$.

Minimizing total notional at a given timestep $t$ is a solved problem, a paper that discusses the impact of solving this problem (derivatives portfolio compression) can be found [here](https://poseidon01.ssrn.com/delivery.php?ID=173022101119066104069084024124124064057072038035075028088075127101004122006005024111124122127028018042026073119104019029013097060013004075058101117086083074115000080085079001122091083005103114006027025067087001080089110082065023117022074089030116069073&EXT=pdf).

The precise problem that a intend to allow a reinforcement agent to learn how to solve is this:

If edges (derivative contracts) that arrive between banks due to a pattern determined by banks within a given financial system, can an agent to learn minimize notional excess over all periods? This can be formalized as solving for:

$$\min \sum_{t \in T} \sum_{\forall i,j \in |B|} w_t(b_i,b_j)$$

Where $w_t(b_i,b_j)$ is a function that returns the weight $w_t$ of the edge defined as a derivative contract relationship between bank $i$ and bank $j$ at timestep $t$.

## Reinforcement Learning Problem

The corresponding reinforcement learning problem is very similar to the problem statement above, but with a few simplifications. They are outlined below:

1. We will only be attempting to solve for the minimum amongst the top 10 most connected nodes, based on initial analysis, one can see that there are very few very connected nodes, so this is a fair simplification for the more general problem. A histogram is shown below:

1. For the early stages of this project, we will only aim to keep total notional below a certain percentage of what I define as counterfactual total notional, which is the total notional that would be observed given no action on a derivatives contract graph $G$, as of right now, this percentage of notional is set at 60%

## Reward Structure

The reward structure of this problem is quite simple, our agent will recieve a reward of $-1$ for every time step it spends with total notional above the given threshold and positive 1 otherwise. Below is the "\_get\_reward" fucntion in our OpenAi Gym environment.
## Action Structure

As previously mentioned, minimizing the total notional amongst a set of banks $B$ and their respective edges $E$ while maintaining the amount owed from each bank to every other is a solved problem. It can be solved using the following linear programming problem:

\begin{equation} \label{eq1}
\begin{split}
\text{minimize } & \hat{u} \cdot e' \\
\text{subject to } & Qe' = v \\
& 0 \leq e' \leq e
\end{split}
\end{equation}

Where $\hat{u}$ is a vector of all 1's, $e'$ is the set of edges that would minimize total notional, $Q$ is an incidence matrix of our derivatives contract graph $G$ and $v$ is a vector representation of the total degree of each bank (which represents underlying value on each banks balance sheet).

To figure out which cycles can be compressed in order achieve the minimum notional solution to the above linear programming problem, one could simply solve the problem, and subtract the solution of the problem from the original adjacency matrix. I denote this as the "critical adjacency matrix". It is composed entirely of "critical cycles" or non-overlapping cycles that must be removed in order to completely conservatively compress a credit derivatives network graph. Say $n$ is the number of critical cycles. Action $n$ of our agent corresponds to subtracting the first $n$ critical cycles from the current adjacency matrix (the adjacency matrix upon which the agent acts (The critical cycles are ordered by the excess notional removed when they are compressed).

## Observation Structure

### State Observations

The observation structure is a bit more complicated, the agent observes the state it exists in which is the culmination of actions it can take, and contracts that have arrived during during the current episode. After it makes a specific action, the agent sees an adjacency matrix repesenting derivatives contracts that have originated up to timestep $t$ between the top $n$ banks with the highest node centrality. As an adjacency matrix is a very noisy way to view and try to discern patterns within a derivatives contract graph, I have looked quite seriously into using the first $k$ singular values of the adjacency matrix to derive a rank $k$ representation of the adjacency matrix.

### Action Observations

In addition, the agent observes the critical cycles of the adjacency matrix representing the culmination of derivatives contracts up to timestep $t$, these critical cycles are calculated by utilizing a conservative compression linear programming algorithm to find the graph that minimizes the notional excess of our derivatives contract graph. I then subtract this graph from the "current adjacency matrix" as described in "State Observations". The cycles of this graph consist entirely of "critical cycles" or cycles that should be removed in order to miniminze excess notional in the derivatives contract graph. Given $n$ critical cycles, our agent can choose between action $ 0 < i \leq n$ where choosing action $i$ runs the following algorithm, as described in (D’Errico and Roukny 2018, for the first $i$ critical cycles (The critical cycles are ordered by the excess notional removed when they are compressed):

## Reward Observation

THe last observation to touch on is the observation that allows the agent to best discern how close it is to an optimal reward. Every timestep, our agent recieved information stating its excess\_percent which is the percent of counterfactual notional that is currently achieved by the current adjacency matrix. This is the same metric that will decide if an agent will recieve a reward of positive or negative 1 during a given timestep.

\_get\_state logic is displayed below:

## Results
Unfortunately, due to the time constraints of the semester, I was unable to build a deep learning agent to train within in the OpenAI Gym I intended to build. However, I was able to build the gym environment and to test various policies on it. For these tests, I used the 10 most connected banks and I used the timesteps of the 57 weeks from January 5th, 1999 to Febuary 8th, 2000. I tested 3 policies:
1. A policy in which the agent compresses all critical cycles left in the derivatives market's graph. At every timestep, this policy calculates the critical cycles of the graph and compresses all of them.
1. A policy in which the agent compresses 500 critical cycles left in the derivatives market's graph. At every timestep, this policy calculates the critical cycles of the graph and compresses all of them.
1. A policy in which the agent compresses a random number of cycles 

Below are the eliminate notional at each time step by each policy. The minimum notional achieved by the random policy uses an average of 100 random polices. The red line, denoted "minimum notional" calculated as the the minimum notional achieved if at the last time step, the algorithm described in Roukny and D'Errico is run. This is the minimum possible excess percent utilizing conservative compression. A compression policy can be evaluated by its distance towards this metric, which for this time interval, and this number of banks is about 44%.