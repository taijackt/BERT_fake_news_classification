import yaml
import pandas as pd
from dataset import FakeNewDataset, create_mini_batch
from train import get_model, get_prediction, train_one_epoch
from preprocess import preprocess_dataframe, read_configs
import torch
from torch.utils.data import DataLoader
import os

def main():
    # read configs
    configs = read_configs()

    # preprocess dataset
    train_df = pd.read_csv(configs["Data_Path"]["train_csv_path"])
    test_df = pd.read_csv(configs["Data_Path"]["test_csv_path"])
    train_df = preprocess_dataframe(train_df, "train",configs)
    test_df = preprocess_dataframe(test_df, "test",configs)

    # Get dataloader
    trainset = FakeNewDataset("train", train_df, configs["Label_map"])
    trainloader = DataLoader(dataset=trainset, batch_size=configs["Training_config"]["batch_size"], collate_fn=create_mini_batch)
    
    # Prepare the model
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model = get_model()
    model = model.to(device)

    # Get the optimizer ready
    optimizer = torch.optim.Adam(model.parameters(), lr=float(configs["Training_config"]["learning_rate"]))

    epochs = configs["Training_config"]["epochs"]

    # Start Training
    for epoch in range(epochs):
        train_one_epoch(model=model, device=device, dataloader=trainloader, optimizer=optimizer);

    # Save model
    save_path = configs["Training_config"]["saving_path"]
    if not (os.path.exists(save_path)):
        os.mkdir(save_path)
        print(f"{save_path} is created.")
    output_model_name = os.path.join(save_path, "output_model.pth")
    torch.save(model.state_dict(), output_model_name)
    print(f"Model is saved at {output_model_name}")

if __name__ == "__main__":
    main()