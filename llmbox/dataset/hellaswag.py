import re

from .multiple_choice_dataset import MultipleChoiceDataset


class Hellaswag(MultipleChoiceDataset):
    """The dataset of hellaswag.

    HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers et al., 2019)
    Hellaswag is a new dataset for commonsense NLI. The paper was published at ACL2019.

    Example:
        'activity_label': 'Roof shingle removal',
        'ctx_a': 'A man is sitting on a roof.',
        'ctx_b': 'he',
        'ctx': 'A man is sitting on a roof. he',
        'endings': ['is using wrap to wrap a pair of skis.',
                    'is ripping level tiles off.',
                    "is holding a rubik's cube.",
                    'starts pulling up roofing on a roof.'],
        'label': '3'
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("Rowan/hellaswag",)

    @staticmethod
    def preprocess(text):
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/a3e56afeab01d1a847ecf37cfed35ffeec2c0150/lm_eval/tasks/hellaswag/utils.py#L6
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text.strip()

    def format_instance(self, instance):
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/a3e56afeab01d1a847ecf37cfed35ffeec2c0150/lm_eval/tasks/hellaswag/utils.py#L15
        source = self.preprocess(
            instance["activity_label"] + ": " + instance["ctx_a"] + " " + instance["ctx_b"].capitalize()
        )

        label2text = {i: " " + self.preprocess(instance["endings"][i]) for i in [0, 1, 2, 3]}
        options = [label2text[option] for option in [0, 1, 2, 3]]
        return dict(
            source=source,
            source_postfix="\nAnswer:" if self.args.ranking_with_options else "",
            target_idx=int(instance["label"]),
            options=options,
        )

    @property
    def references(self):
        return [int(instance["label"]) for instance in self.evaluation_data]
