from diffusers import FluxPipeline
from PIL import Image
import torch


def main():
    # Load the FluxPipeline with the specified model ID
    pipe = FluxPipeline.from_pretrained(
        "/opt/liblibai-models/user-workspace2/model_zoo/FLUX.1-dev",
        torch_dtype="auto",
        dtype=torch.bfloat16,
    )
    

    # Move the pipeline to GPU if available
    pipe.to("cuda:4")

    # Generate an image using the pipeline
    image = pipe(
        prompt="A beautiful landscape with mountains and a river",
        num_inference_steps=28,
        guidance_scale=2.5,
    ).images[0]

    # Save the generated image
    image.save("generated_image.png")

if __name__ == "__main__":
    main()