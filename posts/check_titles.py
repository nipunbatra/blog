#!/usr/bin/env python3
"""
Script to check for missing titles in blog posts (.ipynb and .md files)
"""

import json
import os
import yaml
import re
from pathlib import Path

def check_notebook_title(file_path):
    """Check title in Jupyter notebook"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Check first few cells for metadata
        for i, cell in enumerate(notebook.get('cells', [])[:5]):  # Check first 5 cells
            if cell.get('cell_type') == 'raw':
                source = ''.join(cell.get('source', []))
                if source.strip().startswith('---'):
                    # Parse YAML frontmatter
                    try:
                        # Extract YAML content between --- markers
                        yaml_match = re.search(r'^---\s*\n(.*?)\n---', source, re.DOTALL)
                        if yaml_match:
                            yaml_content = yaml_match.group(1)
                            metadata = yaml.safe_load(yaml_content)
                            if isinstance(metadata, dict):
                                title = metadata.get('title')
                                if title is None:
                                    return "NO_TITLE_FIELD"
                                elif title == "" or title is None:
                                    return "EMPTY_TITLE"
                                else:
                                    return title
                    except yaml.YAMLError:
                        continue
        
        return "NO_METADATA_FOUND"
    except Exception as e:
        return f"ERROR: {str(e)}"

def check_markdown_title(file_path):
    """Check title in Markdown file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for YAML frontmatter
        if content.strip().startswith('---'):
            # Extract YAML frontmatter
            yaml_match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
            if yaml_match:
                try:
                    yaml_content = yaml_match.group(1)
                    metadata = yaml.safe_load(yaml_content)
                    if isinstance(metadata, dict):
                        title = metadata.get('title')
                        if title is None:
                            return "NO_TITLE_FIELD"
                        elif title == "" or title is None:
                            return "EMPTY_TITLE"
                        else:
                            return title
                except yaml.YAMLError:
                    return "YAML_PARSE_ERROR"
        
        return "NO_FRONTMATTER"
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    posts_dir = Path("/Users/nipun/git/blog/posts")
    
    # Find all .ipynb and .md files
    notebook_files = list(posts_dir.glob("*.ipynb"))
    markdown_files = list(posts_dir.glob("*.md"))
    
    missing_titles = []
    empty_titles = []
    has_titles = []
    errors = []
    
    print("Checking notebook files...")
    for nb_file in notebook_files:
        title_result = check_notebook_title(nb_file)
        if title_result == "NO_TITLE_FIELD":
            missing_titles.append(str(nb_file))
        elif title_result == "EMPTY_TITLE":
            empty_titles.append(str(nb_file))
        elif title_result == "NO_METADATA_FOUND":
            missing_titles.append(str(nb_file))
        elif title_result.startswith("ERROR:"):
            errors.append((str(nb_file), title_result))
        else:
            has_titles.append((str(nb_file), title_result))
    
    print("Checking markdown files...")
    for md_file in markdown_files:
        title_result = check_markdown_title(md_file)
        if title_result == "NO_TITLE_FIELD":
            missing_titles.append(str(md_file))
        elif title_result == "EMPTY_TITLE":
            empty_titles.append(str(md_file))
        elif title_result == "NO_FRONTMATTER":
            missing_titles.append(str(md_file))
        elif title_result.startswith("ERROR:"):
            errors.append((str(md_file), title_result))
        else:
            has_titles.append((str(md_file), title_result))
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nFILES WITH NO TITLE FIELD OR METADATA ({len(missing_titles)}):")
    for file_path in sorted(missing_titles):
        print(f"  - {file_path}")
    
    print(f"\nFILES WITH EMPTY TITLES ({len(empty_titles)}):")
    for file_path in sorted(empty_titles):
        print(f"  - {file_path}")
    
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for file_path, error in errors:
            print(f"  - {file_path}: {error}")
    
    print(f"\nFILES WITH TITLES ({len(has_titles)}):")
    for file_path, title in sorted(has_titles):
        print(f"  - {os.path.basename(file_path)}: {title}")
    
    print(f"\nTotal files checked: {len(notebook_files) + len(markdown_files)}")
    print(f"Files missing titles: {len(missing_titles) + len(empty_titles)}")

if __name__ == "__main__":
    main()