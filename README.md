# CartPole - Q-Learning with OpenAI Gym
## About
In this repo I will try to implement a reinforcement learning (RL) agent using the [Q-Learning](https://en.wikipedia.org/wiki/Q-learning)  algorithm. 

## Environment
The code is written and tested in the following environment:
- Ubuntu 18.04
- Python 3.7.0 
- Conda 4.5.11
- OpenAI Gym 0.10.9

## Setup
1. Make sure you have a similar environment as above. OS doesn't matter unless you are running windows (OpenAI Gym doesn't support Windows). You can find information for installing Conda [here](https://conda.io/docs/user-guide/install/index.html) and OpenAI Gym [here](https://gym.openai.com/).
2. Clone this repo to your computer:
   ```bash
   $ git clone https://www.github.com/eight0153/CartPole.git
   ```
3. Cd into the repo directory:
    ```bash
    $ cd CartPole
    ```
4. Run the code:
    ```bash
    $ python main.py
    ```
5. Done!

## Discussion
### Observation Space
The observation space for this problem is described as follows:
```
Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
```

As we can see there are four continuous random variables: cart position, cart velocity, pole angle, pole velocity at tip. This poses an issue for the Q-Learning agent because the algorithm works on a lookup table and it is impossible to maintain a lookup table of all continuous values in a given range. This is because, in theory at least, there exists an infinite number of intermediate values between any two real numbers. When working with floating point numbers on computers there is an upper bound on the number of representable floating point numbers, but even with single precision floats this number is large that keeping a lookup table for all such values would be infeasible. So in short, we need discrete values for the algorithm to be usable.

So how do we convert the continuous observation space into a discrete input space? The answer is bucketing - a method of discretising data, perhaps also known as binning (which is used in histograms). This is done by dividing up the range of values the variable may take into _n_ evenly sized parts. The size of each bucket can be found through the following equation: 

_( |min(x)| + |max(x)| ) / n_ 

where 
- x is the continuous value we want to discretise
- |a| is the absolute value of some arbitrary value
- min(x) is the mininum value in the domain of x
- max(x) is the maximum value in the domain of x
- n is the number of buckets.

Then finding the corresponding bucket for a given value is an iteritave algorithm as such:
<pre>
<b>algorithm</b> find bucket <b>is</b>
    <b>input</b>: observation value x, 
           minimum value of x min_x, 
           maximum value of x max_x, 
           number of buckets n
    <b>output</b>: the bucket of x such that 0 ≤ bucket ≤ n, or -1 if for some reason the bucket was not found (e.g. x was outside the interval [x_min, x_max]])

    bucket size := (abs(min_x) + abs(max_x)) / n

    <b>for each</b> bucket <b>in</b> [0, n) <b>do</b>
        bucket_lower := min_x + bucket × bucket size
        bucket_high := min_x + (bucket + 1) × bucket size

        <b>if</b>  bucket_low ≤ x < bucket_high <b>do</b>
            <b>return</b> bucket

    <b>if</b> x = max_x <b>do</b>
        <b>return</b> n

    <b>return</b> -1
 </pre>

 With this we can 'discretise' the continuous observation space. If we chose _n_=100, i.e. 100 buckets, the values in the observation space would be represented as an integer in the interval [0, 100]. With this we can now easily maintain a lookup table for our Q-values since each variable will have _n_ discrete states.
 
 ### Q-Values and the Lookup Table
 The agent in a Q-learning algorithm uses a lookup table of Q-values. Each row in the table is a state (a particular observation of the world) and each column is an action. Each cell in the lookup table is a Q-value, a value representing how good a given action is for a given state. Therefore, the lookup table is a function f: s, a → q, where s is a state, a is an action, and q is the resultant Q-value.

 Now one issue with this RL problem is that the there are a huge number of possible states, even with only four variables. Luckily with the above bucketing method this has been largely mitigated. However, even if we chose n_buckets = 100, we would have approx. 100^4 = 100,000,000 unique states. Now in terms of Big O notation the number of states grows at a rate of O(N^4), so any increase in n_buckets results in a exponential increase in the number of unique states. This is particularly bad news when we want to store our lookup table. If we were to have a full lookup table which with n_buckets = 100 its dimensions would be 100,000,000 x 2. Assuming we are using 32-bit floats we would end up using 4 bytes * 200,000,000 = 800 MB. Increasing n_buckets by a factor of two (i.e. n_buckets = 200) would result in a lookup table that uses a whole 12.8 GB. So we can see how this could get out of hand very quickly.

 Now the whole reason why you would want to use a lookup table is because it allows O(1) constant time insertion and retrieval, which we will be doing a lot of in RL. However, as demonstrated above, we do not want to store the entire table when we can afford to do so. So my approach has been to create the lookup table as needed (deferred initialisation). I start with an empty dictionary and whenever I need a Q-value that does not yet exist the necessary values are initialised. Now while this does avoid a large lookup table initially, it will eventually get larger and larger as the agent is trained up. One method to mitigate this would be make the state space even smaller (i.e. choosing a smaller value for n_buckets), however this may affect the agent's performance negatively. Now as the lookup table can be thought of as the function f: s, a → q, it is not too much of a logical leap to think that we could use a artificial neural network (ANN) to approximate this function since ANNs are known universal function approximators. In fact this practice is quite common, and when an ANN is used to approximate the Q-values in a Q-Learning agent it is often referred to as 'Deep Q Learning', or DQN for short.  

 ### Exploration vs. Exploitation
 Exploration vs. exploitation is an important concept when it comes to RL. Exploration is about trying out new things in hope that they give a better result than other, already known, aopproaches. Exploitation is about exploiting existing knowledge in order to give the best result possible. It is important to balance both of these aspects to ensure that our agent explores a large range of policies. If our agent was to go full exploitation and never explore, it would learn one policy and  be stuck with a sub-optimal policy. If our agent was to only explore, it would be akin to random behaviour and would not learn. As an analogy for no exploration, it would sort of be like learning about insertion sort and just stopping there and not bothering to learn about any of the other sorting algorithms - insertion sort is great and all, but it may not be the optimal sorting algorithm. So we need some way of ensuring an optimal balance between exploration and exploitation.

 The most intuitive approach to deciding on what action to take would be to take a greedy aproach, i.e. just take whatever action seems the best straight away. Of course there is no exploration in that, and it may prevent our agent from learning a more optimal strategy. This is exactly what happens in this cartpole problem if you simply take the action with the highest Q-value. There are quite a few methods for balancing exploration and exploitation, the most notable are epsilon-greedy, Boltzmann exploration, Pursuit, and UCB-1. These are explored in the paper [Comparing Exploration Strategies for Q-learning in
Random Stochastic Mazes](http://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/Exploration_QLearning.pdf). For my implementation I will be going with UCB-1. A brief explanation of the algorithm would be that you keep a count of how many times you take an action for a given state, and when deciding which action to take you assign each action a bonus for actions with a low relative count so as to promote exploration.