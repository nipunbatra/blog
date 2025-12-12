import nbformat as nbf

b = nbf.v4.new_notebook()

# Metadata
b.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.8.5"
    }
}

# Raw cell for Quarto header
raw_header = """---
title: "Sequential vs Batch Processing with Gemini API"
author: "Nipun Batra"
date: "2025-12-12"
categories: [LLM, Gemini, Performance, Batching, Python]
format:
  html:
    code-fold: false
    toc: true
---"""

b.cells.append(nbf.v4.new_raw_cell(raw_header))

# Intro
text_intro = """# Introduction

When processing large datasets of images with Multimodal LLMs, latency can become a bottleneck. In this post, we explore the performance difference between **sequential** processing (one image at a time) and **concurrent/batch** processing (multiple images in parallel) using the Gemini API.

We will use a simple task: counting the number of people in an image.
"""
b.cells.append(nbf.v4.new_markdown_cell(text_intro))

# Setup
code_setup = """import os
import time
import asyncio
from google import genai
from PIL import Image
import io

# Setup the client
# Ensure GEMINI_API_KEY is set in your environment
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_ID = "gemini-2.0-flash-exp" # Or gemini-1.5-flash
"""
b.cells.append(nbf.v4.new_code_cell(code_setup))

# Data Loading
code_data = """# Load the image
image_path = "bus.jpg" # Using a sample image from the blog
pil_image = Image.open(image_path)

# Create a dataset of 10 identical images for benchmarking
BATCH_SIZE = 10
images = [pil_image] * BATCH_SIZE
prompt = "Count the number of people in this image. Return just the integer number."

print(f"Benchmarking with {BATCH_SIZE} images.")
display(pil_image.resize((300, 200)))
"""
b.cells.append(nbf.v4.new_code_cell(code_data))

# Sequential
text_seq = """## Sequential Processing

First, let's process the images one by one in a simple loop. This is the easiest way to write code but often the slowest.
"""
b.cells.append(nbf.v4.new_markdown_cell(text_seq))

code_seq = """def run_sequential(images, prompt):
    results = []
    start_time = time.time()
    
    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}...", end="\r")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[img, prompt]
        )
        results.append(response.text)
        
    end_time = time.time()
    return end_time - start_time, results

print("Starting sequential processing...")
seq_time, seq_results = run_sequential(images, prompt)
print(f"\nSequential processing took: {seq_time:.2f} seconds")
print(f"Average time per image: {seq_time/len(images):.2f} seconds")
"""
b.cells.append(nbf.v4.new_code_cell(code_seq))

# Batch/Concurrent
text_batch = """## Batch (Concurrent) Processing

Now, let's use Python's `asyncio` to send multiple requests at the same time. The Gemini API supports concurrent requests, which can significantly reduce the total time.
"""
b.cells.append(nbf.v4.new_markdown_cell(text_batch))

code_batch = """async def process_one_async(client, img, prompt):
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=[img, prompt]
    )
    return response.text

async def run_batch(images, prompt):
    start_time = time.time()
    
    # Create tasks for all images
    tasks = [process_one_async(client, img, prompt) for img in images]
    
    # Run them concurrently
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    return end_time - start_time, results

print("Starting batch processing...")
# In a Jupyter notebook, await works directly at the top level
batch_time, batch_results = await run_batch(images, prompt)
print(f"Batch processing took: {batch_time:.2f} seconds")
print(f"Average time per image: {batch_time/len(images):.2f} seconds")
"""
b.cells.append(nbf.v4.new_code_cell(code_batch))

# Comparison
text_compare = """## Comparison

Let's compare the results.
"""
b.cells.append(nbf.v4.new_markdown_cell(text_compare))

code_compare = """speedup = seq_time / batch_time
print(f"Speedup: {speedup:.2f}x")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.bar(['Sequential', 'Batch (Async)'], [seq_time, batch_time], color=['orange', 'skyblue'])
plt.ylabel('Total Time (seconds)')
plt.title(f'Processing {BATCH_SIZE} Images: Sequential vs Batch')
plt.show()
"""
b.cells.append(nbf.v4.new_code_cell(code_compare))

# Save
with open('posts/2025-12-12-gemini-batch-vs-sequential.ipynb', 'w') as f:
    nbf.write(b, f)

print("Notebook created successfully.")
