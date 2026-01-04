---
title: "Terminal Optimization: A 2026 Refresh"
author: "Nipun Batra"
date: "2026-01-03"
categories:
  - terminal
  - productivity
  - macos
  - tools
description: "A comprehensive terminal optimization covering security fixes, modern tooling, and performance improvements"
toc: true
---

# Terminal Optimization: A 2026 Refresh

After years of accumulating cruft in my terminal configuration, I finally did a complete audit and optimization. Here's what I found and fixed.

## The Problems

A quick audit of my `.zshrc` revealed several issues:

### 1. Exposed API Keys (Critical!)

```bash
# DON'T DO THIS - keys visible to anyone with file access
export GEMINI_API_KEY="sk-..."
export GOOGLE_API_KEY="sk-..."
```

API keys sitting in plaintext in a dotfile that might get committed to a dotfiles repo or shared. Not good.

### 2. Dead Code from Deprecated Tools

```bash
# Fig was sunset in 2023 - these do nothing
[[ -f "$HOME/.fig/shell/zshrc.pre.zsh" ]] && source "$HOME/.fig/shell/zshrc.pre.zsh"
```

Fig (now Amazon Q) was deprecated, but these references were still cluttering the config.

### 3. Redundant Conda Initialization

I had conda init blocks in both `.zshrc` (for mambaforge) and `.bash_profile` (for miniconda3). Since switching to `uv` for Python environment management, neither was needed.

### 4. Missing Tool Integrations

I had installed modern CLI tools but never configured them:
- `fzf` - installed but not integrated
- `zoxide` - installed but not activated
- `bat`, `fd`, `eza` - not aliased

### 5. Slow Prompt

The agnoster theme in oh-my-zsh, while pretty, adds noticeable startup latency.

---

## The Solutions

### Secure API Key Management

Created a separate secrets file with restricted permissions:

```bash
# ~/.zsh_secrets (chmod 600)
export GEMINI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

Then in `.zshrc`:

```bash
[ -f ~/.zsh_secrets ] && source ~/.zsh_secrets
```

The secrets file should be:
- Added to `.gitignore_global`
- Set to mode `600` (owner read/write only)
- Never committed to version control

### Starship Prompt

Replaced oh-my-zsh's agnoster theme with [Starship](https://starship.rs/):

```bash
brew install starship
```

In `.zshrc`:

```bash
ZSH_THEME=""  # Disable oh-my-zsh theme
eval "$(starship init zsh)"
```

Starship is:
- Written in Rust (fast)
- Cross-shell compatible
- Highly configurable
- Shows only what's relevant

My `~/.config/starship.toml`:

```toml
add_newline = false
command_timeout = 1000

format = """
$directory\
$git_branch\
$git_status\
$python\
$cmd_duration\
$line_break\
$character"""

[character]
success_symbol = "[❯](bold green)"
error_symbol = "[❯](bold red)"

[directory]
truncation_length = 3
truncate_to_repo = true

[git_status]
ahead = "⇡${count}"
behind = "⇣${count}"
staged = "+${count}"
modified = "!${count}"
```

### Modern CLI Tool Aliases

Replaced legacy commands with modern alternatives:

```bash
# In ~/.zshrc

# Modern ls with icons and git status
alias ls='eza --icons'
alias ll='eza -la --icons --git'
alias lt='eza --tree --level=2 --icons'

# Better cat with syntax highlighting
alias cat='bat --paging=never'
```

Tools installed:
- `eza` - modern `ls` replacement with git integration
- `bat` - `cat` with syntax highlighting
- `fd` - faster, friendlier `find`
- `ripgrep` - faster `grep`
- `zoxide` - smarter `cd`

---

## Before vs After: Real Examples

Let's see these tools in action on a sample Python ML project:

```
dummy-project/
├── config.yaml
├── data/
│   └── readings.csv
├── docs/
├── README.md
├── src/
│   ├── model.py
│   └── train.py      (modified)
└── tests/
    └── test_model.py
```

### `ls` → `eza`: Directory Listing

**Before** (`/bin/ls -la`):
```
total 16
drwxr-xr-x@  9 nipun  staff  288  4 Jan 06:54 .
drwxr-xr-x@  3 nipun  staff   96  4 Jan 06:54 ..
drwxr-xr-x@ 12 nipun  staff  384  4 Jan 07:01 .git
-rw-r--r--@  1 nipun  staff  104  4 Jan 06:54 config.yaml
drwxr-xr-x@  3 nipun  staff   96  4 Jan 06:54 data
drwxr-xr-x@  2 nipun  staff   64  4 Jan 06:54 docs
-rw-r--r--@  1 nipun  staff  235  4 Jan 06:54 README.md
drwxr-xr-x@  4 nipun  staff  128  4 Jan 06:54 src
drwxr-xr-x@  3 nipun  staff   96  4 Jan 06:54 tests
```

**After** (`eza -la --git`):
```
drwxr-xr-x@   - nipun  4 Jan 07:01 -I .git
.rw-r--r--@ 104 nipun  4 Jan 06:54 -- config.yaml
drwxr-xr-x@   - nipun  4 Jan 06:54 -- data
drwxr-xr-x@   - nipun  4 Jan 06:54 -- docs
.rw-r--r--@ 235 nipun  4 Jan 06:54 -- README.md
drwxr-xr-x@   - nipun  4 Jan 06:54 -M src        ← Modified!
drwxr-xr-x@   - nipun  4 Jan 06:54 -- tests
```

Notice the `-M` flag showing the `src/` directory contains modified files. Git status at a glance!

### `ls -R` → `eza --tree`: Tree View

**Before** (`/bin/ls -laR`): Nested, hard to read output spanning many lines.

**After** (`eza --tree --git-ignore`):
```
dummy-project/
├── config.yaml
├── data
│   └── readings.csv
├── docs
├── README.md
├── src
│   ├── model.py
│   └── train.py
└── tests
    └── test_model.py
```

Clean, visual hierarchy. Automatically ignores `.git/` and other gitignored files.

### `cat` → `bat`: Syntax Highlighting

**Before** (`/bin/cat src/model.py`):
```
"""Neural network model for energy prediction."""
import torch
import torch.nn as nn

class EnergyPredictor(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.layers(x)
```

**After** (`bat src/model.py`): Same content but with:
- Syntax highlighting (keywords, strings, classes colored)
- Line numbers
- Git integration (shows modified lines)
- Automatic paging for long files

### `find` → `fd`: Finding Files

**Before** (`find . -name "*.py" -not -path "./.git/*"`):
```
./tests/test_model.py
./src/model.py
./src/train.py
```

**After** (`fd -e py`):
```
src/model.py
src/train.py
tests/test_model.py
```

`fd` is:
- ~5x faster on large codebases
- Ignores `.git/`, `node_modules/`, etc. by default
- Cleaner output (no `./` prefix)
- Colored output in terminal

### `grep` → `rg`: Searching Content

**Before** (`grep -r "torch" . --include="*.py"`):
```
./tests/test_model.py:import torch
./tests/test_model.py:    x = torch.randn(8, 10)
./src/model.py:import torch
./src/model.py:import torch.nn as nn
./src/train.py:import torch
./src/train.py:    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

**After** (`rg "torch"`):
```
README.md:pip install torch pandas numpy
tests/test_model.py:import torch
tests/test_model.py:    x = torch.randn(8, 10)
src/train.py:import torch
src/train.py:    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
src/model.py:import torch
src/model.py:import torch.nn as nn
```

`ripgrep` is:
- ~10x faster than grep
- Respects `.gitignore` automatically
- Searches all file types by default (found the README match!)
- Smart case sensitivity

---

### fzf Integration

Finally configured fzf properly:

```bash
# Install shell integration
$(brew --prefix)/opt/fzf/install

# Configure in .zshrc
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"
export FZF_ALT_C_COMMAND='fd --type d --hidden --follow --exclude .git'
```

Key bindings:
- `Ctrl+T` - fuzzy find files
- `Ctrl+R` - fuzzy search command history
- `Alt+C` - fuzzy cd to directory

### zoxide - Smart Directory Navigation

```bash
eval "$(zoxide init zsh)"
```

Now `z` remembers frequently visited directories:

```bash
z blog      # Jumps to ~/git/blog
z posts     # Jumps to most frequently accessed "posts" directory
zi          # Interactive selection with fzf
```

---

## Ghostty Configuration

Updated my Ghostty terminal emulator config:

```ini
# ~/.config/ghostty/config

# Font
font-family = "JetBrains Mono"
font-size = 16
font-thicken = true

# Appearance
theme = GruvboxDark
background-opacity = 0.95
window-padding-x = 10
window-padding-y = 10
macos-titlebar-style = tabs

# Performance
scrollback-limit = 50000
cursor-style-blink = false

# Keybindings
keybind = super+k=clear_screen
keybind = super+shift+enter=new_split:right
keybind = super+alt+enter=new_split:down

# Shell integration
shell-integration = zsh
shell-integration-features = cursor,sudo,title
```

---

## The Final .zshrc

Here's the cleaned up configuration (~85 lines vs ~160 before):

```bash
# Oh My Zsh
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME=""
DISABLE_UNTRACKED_FILES_DIRTY="true"
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source $ZSH/oh-my-zsh.sh

# Starship prompt
eval "$(starship init zsh)"

# PATH (consolidated, with duplicate removal)
typeset -U PATH
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/.local/share/uv/python/shims:$PATH"
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"

# Tool integrations
[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh
eval "$(zoxide init zsh)"

# Aliases
alias ls='eza --icons'
alias ll='eza -la --icons --git'
alias cat='bat --paging=never'

# uv environment
[ -f ~/.uv/base/bin/activate ] && source ~/.uv/base/bin/activate

# Secrets (API keys)
[ -f ~/.zsh_secrets ] && source ~/.zsh_secrets
```

---

## Results

| Metric | Before | After |
|--------|--------|-------|
| Lines in .zshrc | ~160 | ~85 |
| Shell startup time | ~800ms | ~300ms |
| Exposed secrets | 3 | 0 |
| Dead code blocks | 4 | 0 |
| Modern tools integrated | 0 | 6 |

---

## TL;DR Checklist

If you want to optimize your terminal:

1. **Security audit**: Move all API keys to `~/.zsh_secrets` with `chmod 600`
2. **Remove dead code**: Delete Fig, old conda init, unused PATH entries
3. **Install modern tools**: `brew install starship eza bat fd ripgrep zoxide fzf`
4. **Configure integrations**: fzf keybindings, zoxide init, starship prompt
5. **Add useful aliases**: Replace `ls` with `eza`, `cat` with `bat`
6. **Clean PATH**: Use `typeset -U PATH` to deduplicate
7. **Update terminal emulator**: Ghostty has great defaults, just needs minor config

Happy terminal-ing!
