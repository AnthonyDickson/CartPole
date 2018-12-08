from numpy import full

class ObservationDict:
    """An ObservationDict is a dictionary that maps observations to a list of values.
    Each observation maps up to n_actions number of values, where n_actions is the number
    of actions in the action space.
    """
    def __init__(self, init_value, n_actions, bucketer=None):
        """
        Arguments:
            init_value: the value to initialise cells with.
            n_actions: the number of actions in the problem action space.
            bucketer: the method used to bucket observations. Defaults to None, but if set observations will be 
                      bucketed using this method automatically in get().
        """
        self.table = {}
        self.init_value = init_value
        self.n_actions = n_actions
        self.bucketer = bucketer

    def get(self, observation):
        """Find the cell in the lookup table for the given observation.
        Missing cells are lazily created.
        
        Arguments:
            observation: the observation for the cell to retrieve.
        
        Returns: a reference to the table cell corresonding to the given observation.
        """
        cell = self.table

        if self.bucketer:
            observation = self.bucketer(observation)

        for i, key in enumerate(observation):
            try:
                cell = cell[key]
            except KeyError:
                cell[key] = {} if i != len(observation) - 1 else full(self.n_actions, self.init_value, dtype=float)
                cell = cell[key]

        return cell 

    def __getitem__(self, key):
        return self.get(key)

    def flatten(self):
        for a in sorted(self.table.keys()):
            for b in sorted(self.table[a].keys()):
                for c in sorted(self.table[a][b].keys()):
                    for d in sorted(self.table[a][b][c].keys()):
                        key = [a, b, c, d]
                        row = [key]

                        for value in self.table[a][b][c][d]:
                            row.append(value)

                        yield row

    def to_csv(self):
        result = 'observation, action_0, action_1\n'

        for row in self.flatten():
            result += '{}, {}, {}\n'.format(''.join(map(lambda x: str(x), row[0])), 
                                            row[1], 
                                            row[2])

        return result

    def __str__(self):
        return '\n'.join(map(lambda row: str(row), self.flatten()))
