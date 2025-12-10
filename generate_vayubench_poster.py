#!/usr/bin/env python3
"""
Generate a research poster for VayuBench paper using Gemini 3 Pro Image.
"""

import os
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

# Detailed system prompt for poster generation
poster_prompt = """Create a professional academic research poster for a computer science conference paper titled "VayuBench and VayuChat: Executable Benchmarking and Deployment of LLMs for Multi-Dataset Air Quality Analytics"

LAYOUT & STRUCTURE:
- Conference-style academic poster layout (landscape orientation, 36" x 24" equivalent)
- Clean, modern design with clear visual hierarchy
- Divided into logical sections with clear headings

HEADER SECTION (Top ~20%):
- Title in large, bold sans-serif font: "VayuBench and VayuChat: Executable Benchmarking and Deployment of LLMs for Multi-Dataset Air Quality Analytics"
- Subtitle: "First executable benchmark for air quality analytics + deployed chatbot"
- Authors: Vedant Acharya*, Abhay Pisharodi*, Ratnesh Pasi, Rishabh Mondal, Nipun Batra
- Institution: Indian Institute of Technology Gandhinagar
- Conference: CODS 2025, IISER Pune, India
- Include small institutional logo placeholder on the right

LEFT COLUMN (~30% width):
1. "The Problem" section:
   - Icon: Warning symbol or air pollution symbol
   - Text: "1.6M+ deaths/year in India from air pollution"
   - "Policy decisions require integrating pollution, funding, and demographic data"
   - "Existing tools demand technical expertise or provide shallow insights"

2. "Our Solution" section:
   - Icon: Light bulb or gear symbol
   - Text: "VayuBench: 5,000 natural language queries with verified Python code"
   - "VayuChat: Interactive chatbot for real-time analysis"
   - Small diagram showing: Natural Language → LLM → Python Code → Results

3. "7 Query Categories" section:
   - Small icons for each category:
     • Spatial Aggregation (map icon)
     • Temporal Trends (line chart icon)
     • Spatio-Temporal (calendar + map)
     • Population-Based (people icon)
     • Area-Based (area chart icon)
     • Funding-Related (money icon)
     • Specific Patterns (search icon)

CENTER COLUMN (~40% width):
1. "Benchmark Construction Pipeline" section:
   - 4-step flow diagram with arrows:
     [1. Seed Questions] → [2. Template Design] → [3. Expansion (26K→5K)] → [4. Paraphrasing]
   - Each step in a colored box with brief description

2. "Model Performance" section:
   - Clean bar chart showing exec@1 and pass@1 for top models:
     • Qwen3-Coder-30B (0.99, 0.79) - GREEN bars
     • Qwen3-32B (0.98, 0.78) - BLUE bars
     • Qwen2.5-Coder-14B (0.90, 0.69) - ORANGE bars
     • GPT-OSS-20B (0.88, 0.56) - PURPLE bars
     • Smaller models (grayed out, lower performance)
   - Y-axis: Model names, X-axis: Score (0.0 to 1.0)
   - Legend clearly showing exec@1 vs pass@1

3. "Error Analysis" section:
   - Stacked bar chart showing error types by model:
     • Column errors (RED) - dominant
     • Syntax errors (ORANGE)
     • Name errors (YELLOW)
     • Other errors (GRAY)
   - Shows Column errors are the main bottleneck

RIGHT COLUMN (~30% width):
1. "VayuChat Interface" section:
   - Mockup screenshot showing:
     • Clean chat interface
     • Sample query: "Which cities reduced PM2.5 most relative to their NCAP funding?"
     • Python code output in code block
     • Visualization (bar chart or line graph)
   - Include subtle browser window chrome

2. "Key Datasets" section:
   - 3 icons with labels:
     • CPCB Air Quality (500+ stations, 2017-2024)
     • NCAP Funding (₹9,650 crore, 131 cities)
     • State Demographics (population, area)

3. "Impact & Access" section:
   - QR code placeholder on the left
   - Text on the right:
     • "Try VayuChat: [QR Code]"
     • "GitHub: github.com/sustainability-lab/VayuBench"
     • "HuggingFace Space: SustainabilityLabIITGN/VayuChat"

FOOTER SECTION (Bottom ~10%):
- Key Finding (centered, highlighted box): "Qwen3-Coder-30B achieves 79% pass@1, but column name errors remain the primary challenge for smaller models"
- Contact: vedant.acharya@iitgn.ac.in, nipun.batra@iitgn.ac.in

COLOR SCHEME:
- Primary: Deep blue (#1e3a8a) for headers and key elements
- Secondary: Teal/cyan (#0891b2) for accents and highlights
- Accent: Warm orange (#f97316) for important metrics
- Background: Clean white or very light gray (#f9fafb)
- Text: Dark gray (#1f2937) for body text
- Code blocks: Light gray background (#f3f4f6) with monospace font

TYPOGRAPHY:
- Headers: Bold, sans-serif (like Roboto Bold or Inter Bold)
- Body text: Regular sans-serif (like Roboto or Inter)
- Code: Monospace (like Fira Code or Source Code Pro)
- Clear font size hierarchy: Title (48pt) > Section headers (24pt) > Body (14pt) > Captions (10pt)

DESIGN PRINCIPLES:
- Clean, minimal, academic style (not overly decorative)
- Sufficient white space for readability
- Clear visual hierarchy with consistent spacing
- Professional color scheme suitable for academic conferences
- Icons should be simple, outline-style (not filled/flat)
- Data visualizations should be clear and readable from 6 feet away
- Balance text and visuals (60% visual, 40% text)

VISUAL ELEMENTS:
- Use simple geometric shapes for icons and diagrams
- Charts should have clear labels, gridlines, and legends
- Use subtle shadows or borders to separate sections
- Consistent rounded corners on boxes (8px radius)
- Align all elements to a clear grid system

OUTPUT:
- High resolution poster suitable for printing
- Landscape orientation (3:2 aspect ratio recommended)
- Professional conference poster style
- All text should be clearly readable
- Color scheme should work in both print and digital formats

Generate a polished, publication-ready poster that effectively communicates the key contributions of VayuBench and VayuChat to a technical audience at a computer science conference."""

def generate_poster():
    """Generate the VayuBench research poster."""

    print("Generating VayuBench research poster...")
    print(f"Using model: {IMAGE_MODEL}")
    print("\nThis may take 30-60 seconds...\n")

    try:
        # Generate image using Gemini 3 Pro Image
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=poster_prompt
        )

        # Extract and save the generated image
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data'):
                    # Decode base64 image data
                    image_data = base64.b64decode(part.inline_data.data)
                    generated_img = Image.open(BytesIO(image_data))

                    # Save the poster
                    output_path = "vayubench_poster.png"
                    generated_img.save(output_path, format='PNG', dpi=(300, 300))

                    print(f"✓ Poster generated successfully!")
                    print(f"✓ Saved to: {output_path}")
                    print(f"✓ Image size: {generated_img.size[0]} x {generated_img.size[1]} pixels")

                    # Display image info
                    print(f"\nImage details:")
                    print(f"  Format: {generated_img.format}")
                    print(f"  Mode: {generated_img.mode}")
                    print(f"  Size: {generated_img.size}")

                    return output_path

        print("Error: No image was generated in the response")
        return None

    except Exception as e:
        print(f"Error generating poster: {str(e)}")
        return None

if __name__ == "__main__":
    output_file = generate_poster()

    if output_file:
        print(f"\n{'='*60}")
        print("SUCCESS! Your VayuBench poster is ready.")
        print(f"{'='*60}")
        print(f"\nFile: {output_file}")
        print("\nYou can now:")
        print("  1. Open the image to review")
        print("  2. Print it for the conference")
        print("  3. Share it on social media")
        print("  4. Include it in the paper repository")
    else:
        print("\n" + "="*60)
        print("FAILED: Could not generate poster")
        print("="*60)
        print("\nPlease check:")
        print("  1. GEMINI_API_KEY is set correctly")
        print("  2. You have API quota available")
        print("  3. The model supports image generation")
