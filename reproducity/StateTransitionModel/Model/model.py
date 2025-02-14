import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class STATE_TRANSITION(nn.Module):
    """
        State transition model class
    """
    def __init__(self, epoch, device, learning_rate, hidden_size, model_file):

        super(STATE_TRANSITION, self).__init__()
        # Encoder
        self.gene_number = 978
        self.drug_number = 166
        self.epoch = epoch
        self.my_device = torch.device(device)
        self.learning_rate = learning_rate
        self.model_file = model_file

        self.encoder_hidden_layer1 = nn.Linear(
            in_features=self.drug_number, out_features=self.drug_number
        )

        self.encoder_hidden_layer2 = nn.Linear(
            in_features=self.gene_number, out_features=self.gene_number
        )

        self.encoder_hidden_layer3 = nn.Linear(
            in_features=self.gene_number, out_features=100
        )

        self.encoder_hidden_layer4 = nn.Linear(
            in_features=166, out_features=100
        )

        self.decoder_output_layer1 = nn.Linear(
            in_features=(self.drug_number + self.gene_number), out_features=hidden_size
        )

        self.decoder_output_layer2 = nn.Linear(
            in_features=hidden_size, out_features=self.gene_number
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, drug, ccl):


        activation1 = torch.sigmoid(self.encoder_hidden_layer1(drug))
        activation2 = torch.sigmoid(self.encoder_hidden_layer2(ccl))
        activation = (torch.cat((activation1, activation2), 1))
        reconstructed = torch.sigmoid(self.decoder_output_layer1(activation))
        reconstructed = (self.decoder_output_layer2(reconstructed))

        return reconstructed

    def fit(self, train_loader, valid_loader):

        min_valid_loss = 10000.0
        min_retain_epoch = 0
        for epoch_ind in range(self.epoch):

            train_loss = 0.0
            self.train()
            for batch_drug, batch_ccl, batch_y in train_loader:

                self.optimizer.zero_grad()
                outputs = self.forward(batch_drug, batch_ccl)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            train_loss = train_loss/len(train_loader)

            valid_loss = 0.0
            self.eval()
            with torch.no_grad():

                for batch_drug, batch_ccl, batch_y in valid_loader:
                    outputs = self.forward(batch_drug, batch_ccl)
                    loss = self.loss_fn(outputs, batch_y)
                    valid_loss += loss.item()

                valid_loss = valid_loss/len(valid_loader)

            # early stop
            if valid_loss >= min_valid_loss:
                min_retain_epoch += 1
                if min_retain_epoch >= 5:
                    break
            else:
                min_valid_loss = valid_loss
                min_retain_epoch = 0
                torch.save(self, self.model_file)

            print("epoch %d: train loss %.3f, test loss %.3f" % (epoch_ind, train_loss, valid_loss))

    def predict(self, drug, ccl):

        self.eval()
        drug = torch.tensor(drug, device=self.my_device, dtype=torch.float32)
        ccl = torch.tensor(ccl, device=self.my_device, dtype=torch.float32)
        with torch.no_grad():
            y = self.forward(drug, ccl)

        return y.cpu().detach().numpy()



class MyDataset(Dataset):
    """MyDataset: construct state transition model train dataset class(MyDataset) for pytorch
    """
    def __init__(self, my_device_str, drug, ccl, y):
        my_device = torch.device(my_device_str)
        self.drug = torch.tensor(drug, device=my_device, dtype=torch.float32)
        self.ccl = torch.tensor(ccl, device=my_device, dtype=torch.float32)
        self.y = torch.tensor(y, device=my_device, dtype=torch.float32)

    def __len__(self):
        return len(self.drug)

    def __getitem__(self, index):
        return self.drug[index], self.ccl[index], self.y[index]
