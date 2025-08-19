from diffusers import PEFluxKontextPipeline
from PIL import Image
import torch


def main():
    # Load the PEFluxKontextPipeline with the specified model ID
    pipe = PEFluxKontextPipeline.from_pretrained(
        "/opt/liblibai-models/user-workspace2/model_zoo/FLUX.1-Kontext-dev",
        torch_dtype="auto",
        dtype=torch.bfloat16,
    )
    
    control_img = Image.open("control_img.png").convert("RGB")
    referenced_img = Image.open("referenced_img.png").convert("RGB")

    # Move the pipeline to GPU if available
    pipe.to("cuda:7")

    # Generate an image using the pipeline
    image = pipe(
        prompt="A beautiful landscape with mountains and a river",
        image=control_img,
        reference=referenced_img,
        num_inference_steps=28,
    ).images[0]

    # Save the generated image
    image.save("generated_image.png")

if __name__ == "__main__":
    main()