import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm  

# U-Net архитектура
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_last(dec1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False)):  
        inputs = batch[INPUT_IMAGE_KEY].to(device)
        targets = batch[INPUT_DAMAGE_MASK_KEY].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")
        
    return running_loss / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating", leave=False)): 
            inputs = batch[INPUT_IMAGE_KEY].to(device)
            targets = batch[INPUT_DAMAGE_MASK_KEY].to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx}, Loss: {loss.item()}")
    
    return running_loss / len(loader)

def train_model(model, train_ds, valid_ds, batch_size, num_epochs, device):
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size // 4, num_workers=0, pin_memory=True, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_loss = float('inf')
    best_model = None
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = validate_one_epoch(model, valid_loader, criterion, device)
        
        print(f"Epoch {epoch+1} training loss: {train_loss}")
        print(f"Epoch {epoch+1} validation loss: {valid_loss}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model.state_dict()
    
    print("Training completed")
    return best_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=6, out_channels=5).to(device)

print("Starting training")
trained_model = train_model(model, train_ds, valid_ds, batch_size=16, num_epochs=10, device=device)
print("Training completed")
