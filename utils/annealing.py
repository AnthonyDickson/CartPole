from math import exp

class Annealer: 
    """Anneals a value over time."""

    def __init__(self, k=0.001):
        """Create an annealer that decays a value over time.
        
        Arguments: 
            k: hyperparameter that determines the decay rate/curve.
        """
        self.k = k

   
    def anneal(self, a, t=0):
        """Get the annealed value of a.

        Arguments:
            a: the value to anneal. If None then uses value passed at initialisation.
            t: the timestep.

        Return
        """
        raise NotImplementedError

    def __call__(self, a, t=0):
        return self.anneal(a, t)

class Linear(Annealer):
    def anneal(self, a, t=0):
        return max(0, a - self.k * t)

class Step(Annealer):
    def __init__(self, k=0.001, step_after=10):
        """Create an annealer that decays a value over time.
        
        Arguments: 
            k: hyperparameter that determines the how much the value is decreased each step.
            step_after: how many timesteps before the value should decrease next.
        """
        super().__init__(k)

        self.step_after = step_after
    
    def anneal(self, a, t=0):
        return max(self.k, a - self.k * (t // self.step_after))

class ExponentialDecay(Annealer):
    def anneal(self, a, t=0):        
        return a * exp(-self.k * t)

class TReciprocal(Annealer):
    def anneal(self, a, t=0):
        return a / (1 + self.k * t)