"""Text to image with style embeddings.

Run diffusers pipeline conditioned on
style embeddings.

Usage:
    python -m text_to_image --prompt [prompt] --style [path]

"""

import argparse
import functools
import typing

import diffusers
import torch
import transformers

def optimize(
    image_embedding: torch.Tensor, 
    optimizer: torch.optim.Optimizer, 
    text_encoder: transformers.CLIPTextModel,
    text_ids: torch.Tensor,
    ) -> None:
    """Optimize text encoder.

    Optimize text encoder conditioned with
    image embedding.

    Arguments:
        style_embedding: Image embedding.
        optimizer: Optimizer.
        text_ids: Text ids.

    Returns:
        Returns none.

    """
    text_embedding: torch.Tensor = text_encoder(text_ids)[1]
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True,)
    similarity: torch.Tensor = text_embedding @ image_embedding.T
    loss: torch.Tensor = -similarity.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prompt", help="Text prompt", required=True, type=str,)
parser.add_argument("--style", help="Path to style embedding", required=True, type=str,)
parser.add_argument("--model", help="HuggingFace ID for pre-trained stable diffusion model", required=True, type=str,)
parser.add_argument("--device", help="Device to run model on.", required=True, type=str,)
parser.add_argument("--hftoken", help="HuggingFaces access key", required=True, type=str,)
parser.add_argument("--steps", help="Steps to run conditioning", required=True, type=int,)
arguments = parser.parse_args()

pipeline: diffusers.StableDiffusionPipeline = diffusers.StableDiffusionPipeline.from_pretrained(
    arguments.model, use_auth_token=arguments.hftoken,).to(arguments.device,)

style_embedding: torch.Tensor = torch.load(arguments.style,).float().to(arguments.device,)
style_embedding = style_embedding / style_embedding.norm(dim=-1, keepdim=True,)

text_inputs: torch.Tensor = pipeline.tokenizer(
    arguments.prompt, padding="max_length", max_length=pipeline.tokenizer.model_max_length, return_tensors='pt',)

text_input_ids: torch.Tensor = text_inputs.input_ids.to(arguments.device,)

optimizer: torch.optim.Adam = torch.optim.Adam(pipeline.text_encoder.parameters(), lr=1e-4,)

_optimize: typing.Callable = functools.partial(optimize, style_embedding, optimizer, pipeline.text_encoder,)

list(map(_optimize, [text_input_ids] * arguments.steps))

torch.cuda.empty_cache()
pipeline(arguments.prompt).save("image.jpg")