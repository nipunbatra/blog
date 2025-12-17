#!/usr/bin/env python3
"""
Generic poster/illustration generator using Gemini 3 Pro Image.
Takes a text file with the prompt and generates an image.
"""

import os
import sys
from google import genai
from PIL import Image
from io import BytesIO
import base64

# Initialize Gemini client
if 'GEMINI_API_KEY' not in os.environ:
    raise ValueError(
        "GEMINI_API_KEY not found in environment.\n"
        "Set it with: export GEMINI_API_KEY='your-key'\n"
        "Get your key at: https://aistudio.google.com/apikey"
    )

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
IMAGE_MODEL = "models/gemini-3-pro-image-preview"

def generate_from_prompt_file(prompt_file, output_file=None):
    """
    Generate an image from a prompt text file.

    Args:
        prompt_file: Path to text file containing the prompt
        output_file: Output image filename (auto-generated if None)

    Returns:
        Path to generated image or None if failed
    """

    # Read prompt from file
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_file}' not found")
        return None
    except Exception as e:
        print(f"Error reading prompt file: {str(e)}")
        return None

    # Auto-generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(prompt_file))[0]
        output_file = f"{base_name}_generated.png"

    print(f"Generating image from prompt: {prompt_file}")
    print(f"Output file: {output_file}")
    print(f"Using model: {IMAGE_MODEL}")
    print("\nThis may take 30-60 seconds...\n")

    try:
        # Generate image using Gemini 3 Pro Image
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt
        )

        # Extract and save the generated image
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data'):
                    # Decode base64 image data
                    image_data = base64.b64decode(part.inline_data.data)
                    generated_img = Image.open(BytesIO(image_data))

                    # Save the image
                    generated_img.save(output_file, format='PNG', dpi=(300, 300))

                    print(f"✓ Image generated successfully!")
                    print(f"✓ Saved to: {output_file}")
                    print(f"✓ Image size: {generated_img.size[0]} x {generated_img.size[1]} pixels")

                    # Display image info
                    print(f"\nImage details:")
                    print(f"  Format: {generated_img.format}")
                    print(f"  Mode: {generated_img.mode}")
                    print(f"  Size: {generated_img.size}")

                    return output_file

        print("Error: No image was generated in the response")
        return None

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_poster.py <prompt_file.txt> [output_file.png]")
        print("\nExample:")
        print("  python generate_poster.py vayubench_prompt.txt")
        print("  python generate_poster.py sustainability_lab_prompt.txt")
        print("  python generate_poster.py prompt.txt custom_output.png")
        sys.exit(1)

    prompt_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result = generate_from_prompt_file(prompt_file, output_file)

    if result:
        print(f"\n{'='*60}")
        print("SUCCESS! Your image is ready.")
        print(f"{'='*60}")
        print(f"\nFile: {result}")
        print("\nYou can now:")
        print("  1. Open the image to review")
        print("  2. Print it for presentations/conferences")
        print("  3. Share it on social media")
        print("  4. Include it in your publications")
    else:
        print("\n" + "="*60)
        print("FAILED: Could not generate image")
        print("="*60)
        print("\nPlease check:")
        print("  1. GEMINI_API_KEY is set correctly")
        print("  2. You have API quota available")
        print("  3. The model supports image generation")
        print("  4. The prompt file exists and is readable")
        sys.exit(1)
