def class_counts(rows, target_index):
    counts = dict()
    for row in rows:
        label = row[target_index]
        counts[label] = counts.get(label, 0) + 1
    return counts


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if isinstance(val, int):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = '==' if isinstance(self.value, int) else '>='
        return 'Is Column No.{column} {condition} {value}?'.format(column=self.column,
                                                                   condition=condition,
                                                                   value=self.value)

