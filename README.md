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