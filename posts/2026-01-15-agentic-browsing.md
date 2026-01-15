---
title: "Agentic Browsing with agent-browser"
author: "Nipun Batra"
date: "2026-01-15"
categories: [AI, automation, tools]
description: "A hands-on guide to automating web browsing for AI agents using agent-browser CLI"
toc: true
---

## Introduction

AI agents are increasingly being used for tasks that require web interaction - from research to data collection to form filling. [agent-browser](https://agent-browser.dev/) is a fast CLI tool specifically designed for this purpose. It provides a clean interface for AI agents to interact with websites using an accessibility tree with deterministic element references.

## Installation

```bash
npm install -g agent-browser
agent-browser install  # Install browser binaries
```

## Core Concepts

The key innovation is the **snapshot with refs** - instead of dealing with complex CSS selectors or XPath, agent-browser assigns unique references (`@e1`, `@e2`, etc.) to interactive elements:

```bash
$ agent-browser open "https://news.ycombinator.com"
✓ Hacker News
  https://news.ycombinator.com/

$ agent-browser snapshot -i -c
- link "Hacker News" [ref=e2]
- link "new" [ref=e3]
- link "past" [ref=e4]
- link "comments" [ref=e5]
- link "The URL shortener that makes your links look suspicious" [ref=e12]
- link "35 comments" [ref=e17]
```

Now clicking is trivial: `agent-browser click @e12`

## Real-World Examples

### Example 1: Browsing Wikipedia

Let's explore the IIT Gandhinagar Wikipedia page:

```bash
$ agent-browser open "https://en.wikipedia.org/wiki/IIT_Gandhinagar"
✓ IIT Gandhinagar - Wikipedia
  https://en.wikipedia.org/wiki/IIT_Gandhinagar

$ agent-browser snapshot -i -c | head -30
- link "Jump to content" [ref=e1]
- button "Main menu" [ref=e2]
- searchbox "Search Wikipedia" [ref=e4]
- link "(Top)" [ref=e10]
- link "History" [ref=e11]
- link "Foundation" [ref=e13]
- link "Academics" [ref=e14]
- link "Departments" [ref=e16]
- link "Centres" [ref=e17]
- link "Rankings" [ref=e22]
- link "Campus" [ref=e23]
```

You can take a full-page screenshot:

```bash
$ agent-browser screenshot wiki.png --full
✓ Screenshot saved to wiki.png
```

### Example 2: Searching ArXiv for Research Papers

Search for papers on energy disaggregation (a research area in my lab):

```bash
$ agent-browser open "https://arxiv.org/search/?query=energy+disaggregation&searchtype=all"
✓ Search | arXiv e-print repository
  https://arxiv.org/search/?query=energy+disaggregation&searchtype=all

$ agent-browser snapshot -c -d 3 | head -40
- textbox "Search term or terms" [ref=e6]
- combobox "Field to search" [ref=e9]
- button "Search" [ref=e25]
- main:
  - 'heading "Showing 1–50 of 208 results for all: energy disaggregation" [level=1]'
  - textbox "Search term or terms" [ref=e29]: energy disaggregation
  - paragraph: Revisiting Disaggregated Large Language Model Serving...
  - paragraph: Submitted 14 November, 2025
  - paragraph: Real Time NILM Based Power Monitoring of Identical Motors...
  - paragraph: Submitted 4 January, 2026
```

### Example 3: Extracting Text Content

Get specific text from elements:

```bash
$ agent-browser open "https://news.ycombinator.com"
$ agent-browser get text @e12
The URL shortener that makes your links look as suspicious as possible
```

### Example 4: Form Interaction

Fill and submit forms:

```bash
# Open a search page
$ agent-browser open "https://arxiv.org"
$ agent-browser snapshot -i -c | grep textbox
- textbox "Search term or terms" [ref=e6]

# Fill the search box
$ agent-browser fill @e6 "transformer attention mechanism"

# Submit
$ agent-browser press Enter
```

## Python Integration

You can easily integrate agent-browser with Python scripts:

```python
import subprocess
import json

def run_agent_browser(command):
    """Run agent-browser command and return output."""
    result = subprocess.run(
        f"agent-browser {command}",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# Example: Extract Hacker News headlines
run_agent_browser('open "https://news.ycombinator.com"')
snapshot = run_agent_browser('snapshot -i -c --json')

# Parse and extract headlines
data = json.loads(snapshot)
headlines = [
    elem for elem in data
    if 'link' in str(elem) and 'comments' not in str(elem)
]
```

## Use Cases for Research

### 1. Literature Review Automation
Automatically search and collect paper metadata from ArXiv, Google Scholar, or Semantic Scholar.

### 2. Data Collection
Scrape public datasets, weather data, or government statistics for research projects.

### 3. Conference/Journal Monitoring
Set up automated checks for conference deadlines, paper acceptance notifications, or journal updates.

### 4. Teaching Material Collection
Gather examples, diagrams, or references from educational websites for course preparation.

## Key Commands Reference

| Command | Description |
|---------|-------------|
| `open <url>` | Navigate to URL |
| `snapshot -i -c` | Get interactive elements in compact format |
| `click @ref` | Click element by reference |
| `fill @ref "text"` | Fill input field |
| `get text @ref` | Extract text from element |
| `screenshot [path]` | Take screenshot |
| `close` | Close browser |

## Tips for AI Agent Integration

1. **Use `snapshot -i`** for interactive elements only - reduces noise
2. **Use `snapshot -c`** for compact output - removes empty structural elements
3. **Use `--json`** flag for programmatic parsing
4. **Session isolation**: Use `--session <name>` for parallel browser instances

## Conclusion

agent-browser provides a clean, AI-friendly interface for web automation. The reference-based element selection (`@e1`, `@e2`) makes it ideal for LLM-powered agents that need to interact with websites programmatically.

Links:
- [agent-browser website](https://agent-browser.dev/)
- [GitHub repository](https://github.com/AskUI/agent-browser)
