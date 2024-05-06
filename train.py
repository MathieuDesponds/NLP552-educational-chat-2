from train_utils import (RewardLoss, RewardTrainer, prepare_datasets, load_data, read_args)
from model import (RewardModel, RewardModelConfig)
from transformers import TrainingArguments
import torch
from transformers import RobertaModel, RobertaTokenizer, AdamW
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
import math
from random import shuffle
import sys

import gc

gc.collect()

torch.cuda.empty_cache()



def run(**kwargs):
    """
    arguments :
        verbose : whether we print or not (def: False)
        epochs : number of epochs (def: 100)
        lr : learning rate (def: 1e-5)
        batch size : number of sample in a batch (def: 16)
        warmup_prcnt: (def: 0.3)
    """


    run_args = {
        "VERBOSE" : kwargs.get("VERBOSE", False),
        "EPOCHS" : kwargs.get("EPOCHS", 100),
        "LR" : kwargs.get("LR", 1e-5),
        "BATCH_SIZE" : kwargs.get("BATCH_SIZE", 16),
        "WARMUP_PRCNT" : kwargs.get("WARMUP_PRCNT", 0.3),
        "FFN_HIDDEN_DIMS": kwargs.get("FFN_HIDDEN_DIMS", [128, 64, 32]),
        "FFN_ACTIVATIONS": kwargs.get("FFN_ACTIVATIONS", ["relu", "relu", "relu"]),
        "MODEL_PATH" : kwargs.get("MODEL_PATH", "models_hf")

    }
    ## importing the MODEL
    if(run_args["VERBOSE"]):
        print("1 > loading the model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Chosen device ", device)
    
    ## build the initial LM
    model_name = "roberta-base"
    roberta_model = RobertaModel.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if(run_args["VERBOSE"]):
        print("2 > creating the blank reward model")
    
    ## creating the blank model instance
    reward_model_config = RewardModelConfig(lm_base="roberta-base",
                                            hidden_dims=run_args["FFN_HIDDEN_DIMS"],
                                            activations=run_args['FFN_ACTIVATIONS'])
    reward_model = RewardModel(reward_model_config).to(device)


    ## this is a MOCK part
    if(run_args["VERBOSE"]):
        print("3 > mock creation of the dataset")

    
    dataset = load_data("reward_model_v3_1000_neg.json")
    train_dataset, eval_dataset = dataset["train"], dataset["eval"]

    train_dataset, eval_dataset = prepare_datasets(train_dataset, eval_dataset, tokenizer)
    print(train_dataset)
    ## definition of training arguments
    if(kwargs.get("verbose"), False):
        print("4 > Definition of training arguments")

    ## /!\ DON'T SEEM TO BE USED    
    epochs = run_args["EPOCHS"]
    batch_size = run_args["BATCH_SIZE"]
    lr = run_args["LR"]
    warmup_percent = run_args["WARMUP_PRCNT"]

    total_steps = epochs * math.ceil(len(train_dataset) / batch_size)
    warmup_steps = int(warmup_percent * total_steps)

    optimizer = AdamW(reward_model.parameters(), lr=lr, no_deprecation_warning=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)

    training_args = TrainingArguments(
        output_dir=run_args["MODEL_PATH"],
        num_train_epochs=epochs,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        prediction_loss_only=True, 
        save_total_limit=True
    )

    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        tokenizer=tokenizer
    )
    

    ## training
    if(run_args["VERBOSE"]):
        print("4 > training")
    trainer.train()

    reward_model.save_pretrained(run_args["MODEL_PATH"])
    tokenizer.save_pretrained(run_args["MODEL_PATH"])
    reward_model_config.save_pretrained(run_args["MODEL_PATH"])

if __name__ == "__main__":
    print(sys.argv)
    if(len(sys.argv) == 1):
        print("[DEFAULT RUN]")
        run()
    if(len(sys.argv) > 1 and sys.argv[1] == "test"):
        print("[TEST RUN]")
        print("> proceeding with args :")
        print(">> verbose = True")
        print(">> batch_size = 1")
        run(BATCH_SIZE=1, VERBOSE=True)
    if(len(sys.argv) > 1 and sys.argv[1] == "cold"):
        print("[COLD RUN]")
        print("> proceeding with args :")
        print(">> verbose = True")
        print(">> batch_size = 10")
        run(BATCH_SIZE=10, VERBOSE=True)
    if(len(sys.argv) > 1 and sys.argv[1] == "real"):
        print("[REAL RUN]")
        args = read_args()
        for arg in args:
            print("> proceeding with args :")
            print(arg)
            run(**arg)


