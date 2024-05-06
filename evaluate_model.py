from train_utils import (prepare_single_dataset, load_data)
from model import (RewardModelConfig, RewardModel)
from transformers import (AutoConfig, AutoModel, AutoTokenizer)
from transformers import RobertaTokenizer
import sys
import numpy as np
import torch
import json

path = sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device : ", device)

AutoConfig.register('RewardModel', RewardModelConfig)
AutoModel.register(RewardModelConfig, RewardModel)
tokenizer = AutoTokenizer.from_pretrained(path)
config = AutoConfig.from_pretrained(path)
model = AutoModel.from_pretrained(path, config=config).to(device)


y_plus_rewards = np.array([])
y_minus_rewards = np.array([])

test_dataset = prepare_single_dataset(load_data("reward_model_v3_1000_neg.json")["test"], tokenizer)
print(f"Dealing with {len(test_dataset)} samples")

annotated_data = []
for inputs in test_dataset:
    y_plus = inputs["input_ids"].to(device)
    y_minus = inputs["labels"].to(device)

    y_plus_reward = model(input_ids=y_plus).detach().cpu().numpy()
    y_minus_reward = model(input_ids=y_minus).detach().cpu().numpy()
    y_plus_rewards = np.append(y_plus_rewards, y_plus_reward) ## this is R(Y+)
    y_minus_rewards = np.append(y_minus_rewards, y_minus_reward) ## this is R(Y-)
    
    annotated_data.append({
            "y_plus" : tokenizer.decode(y_plus),
            "y_minus": tokenizer.decode(y_minus),
            "y_plus_reward": y_plus_reward,
            "y_minus_reward": y_minus_reward
        })

rewards_diff = y_plus_rewards - y_minus_rewards

avg_difference = np.mean(rewards_diff)
std_difference = np.std(rewards_diff)

ranking_accuracy = np.mean(y_plus_rewards > y_minus_rewards)

print('The average value of the differences between Y+ rewards and Y-rewards: ', avg_difference)
print('The std of the differences between Y+ rewards and Y-rewards: ', std_difference)

print('Proportion of correct order Y+ and Y-:', ranking_accuracy)
annotated_data = [{"avg_difference" : avg_difference, "std_difference" : std_difference, "ranking_accuracy": ranking_accuracy}]
with open("res_on_" + path + ".json", "w") as f:
        json.dump(annotated_data, f)


