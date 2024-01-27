from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.output_parsers.boolean import BooleanOutputParser

class PromptDataset(Dataset):

    def __init__(self, data, question_transforms=None, answer_transforms=None):
        self.questions = data['question'].tolist()
        self.options = data['options'].tolist()
        self.answer = data['answer'].tolist()
        self.meta_info = data['meta_info'].tolist()

        self.question_transforms = question_transforms
        self.answer_transforms = answer_transforms

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        current_answers = self.options[idx]
        current_correct = self.answer[idx]

        if self.question_transforms:
            for transform in self.question_transforms:
                current_question = transform(current_question)
        if self.answer_transforms:
            for transform in self.answer_transforms:
                current_answers = transform(current_answers)

        return current_question, current_answers, current_correct

class BooleanJsonParser(BooleanOutputParser):

    json_parser = SimpleJsonOutputParser()

    def parse(self, text):
        return_dict = {}
        text = self.json_parser.parse(text)
        for key, value in text.items():
            return_dict[key] = super().parse(value)
        
        return return_dict


