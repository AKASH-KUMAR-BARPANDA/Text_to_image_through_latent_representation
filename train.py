from torch.utils.data import DataLoader
from dataset import customdataset
import torch
from image_decoder import ImageDecoder
from image_encoder import ImageEncoder

""" TO DEVICE """
device = "mps" if torch.mps.is_available() else "cpu"


dataset = customdataset(
    imageDir="/Users/akash/Documents/projects try_on/Latent_model_test/dataset/Images",
    caption="/Users/akash/Documents/projects try_on/Latent_model_test/dataset/captions.txt"
    )

trainloader = DataLoader(dataset,shuffle=True,batch_size=4)
decoder = ImageDecoder().to(device)
encoder = ImageEncoder().to(device)

""" HYPER PARAMETERS"""
EPOCHS = 5
LEARNING_RATE = 1e-3
CRITERION = torch.nn.MSELoss()
OPTIMIZER = torch.optim.Adam(
    list(encoder.parameters())+list(decoder.parameters()),
    lr=LEARNING_RATE)


""" TRAINING """
if __name__ == "__main__":
    
    for epoch in range(EPOCHS):
        for idx,(img,caption) in enumerate(trainloader):
            img = img.to(device)
            img_latent = encoder(img)
            reconstruct_img = decoder(img_latent)
            loss = CRITERION(reconstruct_img,img)
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            if idx % 10000 == 0:
                print(f"EPOCH -> {epoch} || LOSS -> {loss.item():.4f}")

""" SAVE (.pth)"""   
torch.save(
    encoder.state_dict(),
    "encoder.pth"
)

torch.save(
    decoder.state_dict(),
    "decoder.pth"
)