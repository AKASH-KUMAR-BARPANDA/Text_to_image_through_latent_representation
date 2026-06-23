
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch
from image_encoder import ImageEncoder
from dataset import customdataset
from torch.utils.data import DataLoader

""" PROJECTION CLASS """
class TextProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384,512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512,512)
    
    def forward(self,text_embedding):
        output = self.relu(self.fc1(text_embedding))
        return self.fc2(output)

""" LOAD MODEL """
text_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

""" TRAINING Z SPACE """

device = "mps" if torch.mps.is_available() else "cpu"

dataset = customdataset(
    imageDir="/Users/akash/Documents/projects try_on/Latent_model_test/dataset/Images",
    caption="/Users/akash/Documents/projects try_on/Latent_model_test/dataset/captions.txt"
    )

trainloader = DataLoader(dataset,shuffle=True,batch_size=4)
projection = TextProjection().to(device)

""" LOAD AND FREEZE ENCODER PARAMETERS """
encoder = ImageEncoder().to(device)
encoder.load_state_dict(torch.load(
    "/Users/akash/Documents/projects try_on/Latent_model_test/encoder.pth",
      map_location=device))
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False


""" HYPER PARAMETERS """
CRITERION = nn.MSELoss()
LEARNING_RATE = 1e-3
OPTIMIZER = torch.optim.Adam(projection.parameters(),lr=LEARNING_RATE)
EPOCHS = 5

""" GRADIENT CALCULATION"""

if __name__ == "__main__":

    for epoch in range(EPOCHS):
        loss = 0
        for images, captions in trainloader:
            images = images.to(device)

            with torch.no_grad():
                z_img = encoder(images)
                z_text_embedding = (
                    text_model.encode(captions,convert_to_tensor=True).to(device).clone())

            z_text = projection(z_text_embedding)

            loss = CRITERION(z_text,z_img)

            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
        
        if epoch % 1000 == 0:
            print(f"EPOCH ->{epoch} || LOSS -> {loss.item()}")


""" SAVE """
torch.save(
    projection.state_dict(),
    "projection.pth"
)