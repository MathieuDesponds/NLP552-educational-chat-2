import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer, PreTrainedTokenizer
from datasets import Dataset
import json
from torch.utils.data import Dataset
from typing import List, Dict

def load_data(path: str) -> List[Dict]:
    """
        Reads the JSON list file given in argument and output the result
    """
    with open(path, 'rb') as f :
        data = json.load(f)
    return data

def read_args() -> List[Dict[str, str]]:
    """
        Read the file .training_args that contains all the parameters
        for the training runs.
    """
    run_args = []
    with open(".training_args", "r") as argsf:
        full_args = argsf.read()
        runs = full_args.split("RUN:")[1:]
        print(runs)
        for run in runs:
            args = {}
            for spec in run.split("\n")[1:]:
                if(spec != ''):
                    var_name, var_val = spec.split("=")
                    var_val = eval(var_val)
                    args[var_name] = var_val
            run_args.append(args)
    return run_args



def prepare_single_dataset(ds: List[Dict], tokenizer: PreTrainedTokenizer) -> Dataset:
    """
        Prepares the dataset for the training.
            - format the answer in the following way : [CLS] <question> [SEP] <answer> (made by format_question function)
            - tokenizes the results
            - Puts both result in "input_ids" and "labels" to make it pass the Trainer checks
    """
    col_match = {
        "y_plus_answer": "input_ids",
        "y_minus_answer": "labels"
    }
    def format_question(sample):
        x = sample["question"]
        if(sample["choices"] is not None):
            x += " ".join(f"{i}) {choice}" for i, choice in enumerate(sample["choices"]))
        return x

    out_ds = {}
    for col in ["y_plus_answer", "y_minus_answer"]:
        ## max length has been computed in a different notebooks
        out_ds[col_match.get(col, col)] = [torch.tensor(tokenizer.encode("[CLS] " + format_question(sample) + " [SEP] " + sample[col], padding='max_length', truncation=True, max_length=512)) for sample in ds]
    return Dataset.from_dict(out_ds)

def prepare_datasets(train_dataset, eval_dataset, tokenizer):
    return prepare_single_dataset(train_dataset, tokenizer), prepare_single_dataset(eval_dataset, tokenizer)


class RewardTrainer(Trainer):


    def compute_loss(self, model, inputs, return_outputs=False):
        """
            Compute the loss using the custom RewardLoss created under this class
            The loss - log(sig(R(Y+)) - sig(R(Y-)))

            The good answer is stored in input_ids and the bad one is stored in labels
            To make it work properly with the Trainer class
        """
        input_good = inputs["input_ids"]
        input_bad = inputs["labels"]
        output_good = model(input_ids=input_good) ## this is R(Y+)
        output_bad = model(input_ids=input_bad) ## this is R(Y-)
        loss_f = RewardLoss()
        loss = loss_f(output_good, output_bad) ## loss function in inputed in super.__init__()
        return (loss, (output_good, output_bad)) if return_outputs else loss


    def evaluate(self, ignore_keys):
        """
            Computes the average loss over eval samples.

            /!\ /!\ /!\ /!\
            This function is not clean and will fail on CPU runs.
            /!\ /!\ /!\ /!\ 
        """

        overall_eval_loss = 0.0
        for inputs in self.eval_dataset:
            inputs["input_ids"] = torch.tensor(inputs["input_ids"]).unsqueeze(0).to("cuda:0")
            inputs["labels"] = torch.tensor(inputs["labels"]).unsqueeze(0).to("cuda:0")
            overall_eval_loss += self.compute_loss(self.model, inputs).item()

        return overall_eval_loss / len(self.eval_dataset)
    
class RewardLoss(nn.Module):
    ## loss that will be used to train the Reward Model

    def __init__(self):
        super().__init__()

    def forward(self, output1: torch.Tensor, output2: torch.Tensor) -> torch.Tensor:
        ## implement the - log(sig(R(Y+) - R(Y-))) where inputs are R(Y+) and R(Y-)
        return - torch.log(F.sigmoid(output1 - output2)).mean().squeeze(0)
