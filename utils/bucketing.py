class BuckterInterface:
    """Interface for bucketers."""
    def get_bucketed(self, value):
        """Convert a value into a discrete value via bucketing.
        
        Arguments:
            value: the value to discretise.

        Returns: the bucketed value.
        """
        raise NotImplementedError

    def __call__(self, value):
        return self.get_bucketed(value)

class Bucketer(BuckterInterface):
    """Divides a continuous input space into n evenly sized buckets."""

    def __init__(self, lower_bound, upper_bound, n_buckets):
        """Create a bucketer that divides a continuous input space into even parts (buckets).

        Buckets (or bins) a value v into a bucket, where
            v = {x | lower_bound ≤ x ≤ upper_bound}, 
            bucket = {x | 0 ≤ x ≤ n_buckest}.

        The size of a bucket is calculated as:
            (|lower_bound| + |upper_bound|) / n_buckets

        Arguments:
            lower_bound: the lower bound for input space.
            upper_bound: the upper bound for the input space.
            n_buckets: the number of buckets to split the input space into. 
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_buckets = n_buckets

    def get_bucketed(self, value):
        """Convert a value into a discrete value via bucketing.
        
        Arguments:
            value: the value to discretise.

        Returns: the bucketed value.
        """
        step_size = abs(self.lower_bound) / self.n_buckets + abs(self.upper_bound) / self.n_buckets

        for bucket in range(self.n_buckets):
            if self.lower_bound + bucket * step_size <= value < self.lower_bound + (bucket + 1) * step_size:
                return bucket

        if bucket == self.upper_bound:
            return self.n_buckets
        
        return -1

    def __call__(self, value):
        return self.get_bucketed(value)

class MultiBucketer(BuckterInterface):
    """Same as Bucketer except it buckets a vector where each element is a bucketing function.
    
    So if a bucketer is represented as B: x → y, then a MultiBucketer is the vector {B_1: x_1 → y_1, B_2: x_2 → y_2, ... , B_n: x_n → y_n }^T
    """

    def __init__(self, lower_bounds, upper_bounds, n_buckets):
        assert len(lower_bounds) == len(upper_bounds)

        self.n = len(lower_bounds)
        self.n_buckets = n_buckets
        self.bucketers = [Bucketer(lower, upper, n_buckets) for (lower, upper) in zip(lower_bounds, upper_bounds)]

    def get_bucketed(self, values):
        return [bucketer.get_bucketed(value) for (bucketer, value) in zip(self.bucketers, values)]

    def __call__(self, values):
        return self.get_bucketed(values)