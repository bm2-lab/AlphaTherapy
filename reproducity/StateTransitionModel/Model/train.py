import os
import pickle
import numpy as np
from pathlib import Path
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader
from model import STATE_TRANSITION, MyDataset

# Load the config file
config = ConfigParser()
config.read("./model.config", encoding="UTF-8")

# Iterate through all sections except DEFAULT
for section in config.sections():

    if section == "DEFAULT":
        continue

    # Set parameters for the current section
    seed = config.getint(section, "seed", fallback=config.getint("DEFAULT", "seed"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Random Seed for {section}: {seed}")

    # File parameters
    input_file = config.get(section, "input_file", fallback=config.get("DEFAULT", "input_file"))
    model_file = config.get(section, "model_file", fallback=config.get("DEFAULT", "model_file"))
    predict_file = config.get(section, "predict_file", fallback=config.get("DEFAULT", "predict_file"))

    # Model parameters
    hidden_size = config.getint(section, "hidden_size", fallback=config.getint("DEFAULT", "hidden_size"))

    # Training parameters
    device = config.get(section, "device", fallback=config.get("DEFAULT", "device"))
    epoch = config.getint(section, "epoch", fallback=config.getint("DEFAULT", "epoch"))
    batch_size = config.getint(section, "batch_size", fallback=config.getint("DEFAULT", "batch_size"))
    learning_rate = config.getfloat(section, "learning_rate", fallback=config.getfloat("DEFAULT", "learning_rate"))

    print(f"Training configuration for {section}:")
    print(f"Device: {device}, Epochs: {epoch}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    # Load data
    with open(input_file, "rb") as f:
        (
            train_drug, train_x, train_y,
            valid_drug, valid_x, valid_y,
            test_drug, test_x, test_y,
            scaler_control, scale_change, y
        ) = pickle.load(f)

    train_dataset = MyDataset(device, train_drug, train_x, train_y)
    valid_dataset = MyDataset(device, valid_drug, valid_x, valid_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Train model
    model_file_path = Path(model_file).parent
    model_file_path.mkdir(parents=True, exist_ok=True)

    model = STATE_TRANSITION(
        epoch=epoch,
        device=device,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        model_file=model_file
    )
    model = model.to(torch.device(device))
    model.fit(train_loader, valid_loader)

    # Save predictions
    model = torch.load(model_file)
    model.eval()

    pred_y = model.predict(test_drug, test_x)

    with open(predict_file, 'wb') as f:
        pickle.dump([test_y, pred_y], f)

    print(f"Finished training and saved predictions for {section}.\n")
