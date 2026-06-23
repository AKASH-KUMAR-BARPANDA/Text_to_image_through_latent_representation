import torch
from image_encoder import ImageEncoder
from image_decoder import ImageDecoder
from text_encoder import TextProjection,text_model
import cv2


""" DEVICE """
device = "mps" if torch.mps.is_available() else "cpu"

""" LOAD ALL THE MODEL"""
encoder = ImageEncoder().to(device)
decoder = ImageDecoder().to(device)
projection = TextProjection().to(device)

encoder.load_state_dict(torch.load("encoder.pth",map_location=device))
decoder.load_state_dict(torch.load("decoder.pth",map_location=device))
projection.load_state_dict(torch.load("projection.pth",map_location=device))

encoder.eval()
decoder.eval()
projection.eval()

caption = [
    "A brown dog running on a street."
]

embedding = text_model.encode(
    caption,
    convert_to_tensor=True
).to(device)

embedding = embedding.clone()

with torch.no_grad():
    z = projection(embedding)
    generated_image = decoder(z)


img = generated_image.squeeze(0)
img = img.cpu()
img = img.permute(1, 2, 0) # as pytorch store images as (Channel, Height, Width) , where as open cv want as (Height, Width, Channel)
img = img.numpy()
print(img.min(), img.max())
img = img - img.min() # subtract by min so no output will be negative
img = img / (img.max() + 1e-8) # the output will comes between 0 to 1
img = (img * 255).astype("uint8") # converted the output from 0 to 1 -> 0 to 255 as open cv need from 0 to 255 

img = cv2.cvtColor(
    img,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    "generated_image.png",
    img
)