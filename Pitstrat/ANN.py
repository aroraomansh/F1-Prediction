import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MLPModel(nn.Module):
    def __init__(self, circuit_id_max, race_id_max, dropout_prob=0.3, weight_decay=1e-4):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(circuit_id_max + race_id_max + 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5=nn.Linear(32,16) #change
        self.fc6=nn.Linear(16,8)
        self.elu = nn.ELU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(16)
        self.bn6 = nn.BatchNorm1d(8)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.weight_decay = weight_decay  # for optimizer configuration
        self.circuit_id_max=circuit_id_max
        self.race_id_max=race_id_max
    
    def forward(self, circuit_onehot, race_onehot, points_wins):
        x = torch.cat([circuit_onehot, race_onehot, points_wins], dim=1)
        x = self.dropout(self.elu(self.bn1(self.fc1(x))))
        x = self.dropout(self.elu(self.bn2(self.fc2(x))))
        x = self.dropout(self.elu(self.bn3(self.fc3(x))))
        x = self.dropout(self.elu(self.bn4(self.fc4(x))))
        x = self.dropout(self.elu(self.bn5(self.fc5(x))))
        x = self.fc6(x)  # No activation for final layer (raw logits)
        return x


class NN_Eval():
    def __init__(self,model, train_loader, test_loader):
        self.model=model
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.circuit_id_max=self.model.circuit_id_max
        self.race_id_max=self.model.race_id_max
        
    
    def train_model(self ,optimizer, criterion, epochs=100, patience=2000, device='cuda'):
        best_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                circuit_onehot = inputs[:, :self.circuit_id_max]
                race_onehot = inputs[:, self.circuit_id_max:self.circuit_id_max + self.race_id_max]
                points_wins = inputs[:, self.circuit_id_max + self.race_id_max:]

                outputs = self.model(circuit_onehot, race_onehot, points_wins)

                loss = criterion(outputs, targets.argmax(dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}', end=" ")

            # Validation step
            val_loss = self.validate_model( criterion)
            print(f'Validation loss: {val_loss:.4f}')

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0  # Reset counter if improvement is seen
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1} with patience {patience} reached.')
                    break  # Stop training

    def validate_model(self, criterion, device='cuda'):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                circuit_onehot = inputs[:, :self.circuit_id_max]
                race_onehot = inputs[:, self.circuit_id_max:self.circuit_id_max + self.race_id_max]
                points_wins = inputs[:, self.circuit_id_max + self.race_id_max:]

                outputs = self.model(circuit_onehot, race_onehot, points_wins)
                loss = criterion(outputs, targets.argmax(dim=1))
                running_loss += loss.item()

        return running_loss / len(self.test_loader)


class F1Dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Usage example: Include weight decay in your optimizer
# model = MLPModel(circuit_id_max, race_id_max)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=model.weight_decay)
