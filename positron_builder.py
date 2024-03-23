class Value:
    def __init__(self, data, label='', parents=(), _operator=''):
        self.data = data
        self._prev = set(parents)
        self._operator = _operator
        self.label = label
        self.gradient = 0.0 # This is computed in the backwards pass

    def __repr__(self):
        labels = [child.label for child in self._prev]
        if len(labels) == 2:
            parents = (labels[0], labels[1])
        else:
            parents = ()

        return f'Value(data={self.data}, label="{self.label}", parents={parents})'

    def __add__(self, other):
        return Value(
            data = self.data + other.data,
            parents = (self, other),
            _operator = '+'
        )

    def __mul__(self, other):
        return Value(
            data = self.data * other.data,
            parents = (self, other),
            _operator = '*'
        )

def label(value: Value, label: str) -> Value:
    value.label = label
    return value
