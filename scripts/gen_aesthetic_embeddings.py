import argparse
import clip
import glob
from PIL import Image
import torch
import tqdm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--images", help="Path to directory containing images.", required=True, type=str,)
parser.add_argument("--path", help="Path to write the embedding.", required=True, type=str,)
arguments = parser.parse_args()

# Just put your images in a folder inside reference_images/
aesthetic_style = arguments.images
image_paths = glob.glob(aesthetic_style)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


with torch.no_grad():
    embs = []
    for path in tqdm.tqdm(image_paths):
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        emb = model.encode_image(image)
        embs.append(emb.cpu())

    embs = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)

    # The generated embedding will be located here
    torch.save(embs, arguments.path)
