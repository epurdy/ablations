from ablations.core import ModelAblator, TaskHarness


class FirstPronounGenderTask(TaskHarness):
    def __init__(self, model):
        he_tokens = model.tokenizer.encode(" he")
        assert len(he_tokens) == 1
        self.he_index = he_tokens[0]
        she_tokens = model.tokenizer.encode(" she")
        assert len(she_tokens) == 1
        self.she_index = she_tokens[0]

        self.example_text_because = (
            "Now Mary dislikes John, because he kicked her"
        )
        self.example_text_so = (
            "Now Mary dislikes John, so she kicked him"
        )
        
    def get_examples(self):
        return dict(
            because=self.example_text_because,
            so=self.example_text_so,
        )
        
    def get_logit_diff(self, logits, key):
        # Takes in a batch x position x vocab tensor of logits
        if key == 'because':
            return (logits[0, -4, self.he_index] -
                    logits[0, -4, self.she_index])
        elif key == 'so':
            return (logits[0, -4, self.she_index] -
                    logits[0, -4, self.he_index])
        else:
            assert False, f'Unknown key: `{key}`'


class SecondPronounGenderTask(TaskHarness):
    def __init__(self, model):
        him_tokens = model.tokenizer.encode(" him")
        assert len(him_tokens) == 1
        self.him_index = him_tokens[0]
        her_tokens = model.tokenizer.encode(" her")
        assert len(her_tokens) == 1
        self.her_index = her_tokens[0]

        self.example_text_because = (
            "Now Mary dislikes John, because he kicked her"
        )
        self.example_text_so = (
            "Now Mary dislikes John, so she kicked him"
        )

    def get_examples(self):
        return dict(
            because=self.example_text_because,
            so=self.example_text_so,
        )
        
    def get_logit_diff(self, logits, key):
        # Takes in a batch x position x vocab tensor of logits
        if key == 'because':
            return (logits[0, -2, self.her_index] -
                    logits[0, -2, self.him_index])
        elif key == 'so':
            return (logits[0, -2, self.him_index] -
                    logits[0, -2, self.her_index])
        else:
            assert False, f'Unknown key: `{key}`'


def main():
    task1 = FirstPronounGenderTask(model)
    task2 = SecondPronounGenderTask(model)

    ablator = ModelAblator(tasks=[task1, task2])



if __name__ == '__main__':
    main()
