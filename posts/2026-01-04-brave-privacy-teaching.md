---
title: "Brave Browser Setup for Teaching & Presentations"
author: "Nipun Batra"
date: "2026-01-04"
categories:
  - productivity
  - teaching
  - privacy
  - browser
description: "Configure Brave browser to hide personal browsing history during classroom demos and presentations"
toc: true
---

# Brave Browser Setup for Teaching & Presentations

When presenting to students or in meetings, the last thing you want is your personal browsing history, bookmarks, or search suggestions appearing in the URL bar. Here's how to configure Brave browser for distraction-free, privacy-respecting presentations.

## Why Brave?

| Feature | Chrome | Brave |
|---------|--------|-------|
| Built-in ad blocking | No | Yes |
| Built-in tracker blocking | No | Yes |
| Google account required | Effectively yes | No |
| Battery usage | High | Medium |
| Privacy defaults | Poor | Good |

## Quick Setup: Dedicated Teaching Profile

The cleanest solution is a separate browser profile for teaching.

### Create a Teaching Profile

1. Click the **profile icon** (top-right corner)
2. Click **Add Profile**
3. Name it "Teaching" or "Presentations"
4. **Don't sign in** to any accounts
5. Skip all sync options

### Switch Profiles

- **Menu**: Click profile icon → Select profile
- **Keyboard**: `Cmd + Shift + M` (macOS) / `Ctrl + Shift + M` (Windows/Linux)

---

## Essential Extensions

Install these in your Teaching profile:

| Extension | Purpose | Install |
|-----------|---------|---------|
| **Unhook** | Removes YouTube recommendations, comments, sidebar | [Get it](https://unhook.app/) |
| **Blank New Tab** | Clean new tab page, no history/suggestions | Chrome Web Store |
| **uBlock Origin** | Ad blocker (backup to Brave's built-in) | Chrome Web Store |

### Installing Extensions in Brave

1. Go to `brave://extensions/`
2. Enable **Developer mode** (top right)
3. Search Chrome Web Store or drag `.crx` files

---

## Disable URL Bar Suggestions

Even with a fresh profile, Brave shows suggestions as you type. Here's how to disable them completely.

### Method 1: GUI Settings

```
brave://settings/search
```

Turn **OFF**:

- [ ] Autocomplete searches and URLs
- [ ] Show search suggestions
- [ ] Improve search suggestions

Also visit:

```
brave://settings/autofill
```

Turn **OFF**:

- [ ] Save and fill addresses
- [ ] Save and fill payment methods

### Method 2: Flags (Nuclear Option)

```
brave://flags/#omnibox-history-quick-provider
```

Set to **Disabled**, then restart Brave.

### Method 3: Python Script (Reproducible)

For applying settings across multiple machines, use this script:

#### The Patch File

Create `brave_privacy_patch.json`:

```json
{
  "search": {
    "suggest_enabled": false
  },
  "ntp": {
    "num_personal_suggestions": 0
  },
  "omnibox": {
    "local_history_zero_suggest_eager_loading_enabled": false,
    "rich_autocompletion_full_url_enabled": false
  },
  "documentsuggest": {
    "enabled": false
  },
  "dns_prefetching": {
    "enabled": false
  },
  "network_prediction_options": 2,
  "brave": {
    "autocomplete_enabled": false,
    "top_site_suggestions": false
  }
}
```

#### The Apply Script

Create `apply_brave_privacy.py`:

```python
#!/usr/bin/env python3
"""Apply privacy patch to Brave Browser preferences."""

import json
import platform
from pathlib import Path
import shutil
from datetime import datetime


def get_brave_prefs_path() -> Path:
    """Get Brave preferences path for current OS."""
    home = Path.home()
    paths = {
        "Darwin": home / "Library/Application Support/BraveSoftware/Brave-Browser/Default/Preferences",
        "Linux": home / ".config/BraveSoftware/Brave-Browser/Default/Preferences",
        "Windows": home / "AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/Preferences",
    }
    return paths.get(platform.system())


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge patch into base dict."""
    result = base.copy()
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_patch(patch_file: str = "brave_privacy_patch.json"):
    """Apply patch to Brave preferences."""
    prefs_path = get_brave_prefs_path()

    if not prefs_path.exists():
        print(f"Error: Preferences not found at {prefs_path}")
        return

    # Backup
    backup = prefs_path.with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy(prefs_path, backup)
    print(f"Backup: {backup}")

    # Load current prefs and patch
    with open(prefs_path) as f:
        prefs = json.load(f)
    with open(patch_file) as f:
        patch = json.load(f)

    # Merge and save
    merged = deep_merge(prefs, patch)
    with open(prefs_path, 'w') as f:
        json.dump(merged, f, indent=3)

    print("✓ Applied! Restart Brave.")


if __name__ == "__main__":
    apply_patch()
```

#### Usage

```bash
# Close Brave first!
python apply_brave_privacy.py

# Then restart Brave
```

This approach is:

- **Reproducible**: Same settings on any machine
- **Version controlled**: Store in dotfiles repo
- **Minimal**: Only changes what's needed

---

## Clear Existing History

After applying settings, clear existing data:

### GUI Method

Press `Cmd + Shift + Delete` (macOS) or `Ctrl + Shift + Delete` (Windows/Linux):

- [x] Browsing history
- [x] Cached images and files
- [x] Autofill form data
- [ ] Cookies (keep if you want to stay logged in)

### Command Line Method

```bash
# Close Brave first!
BRAVE_DIR=~/Library/Application\ Support/BraveSoftware/Brave-Browser/Default

rm -f "$BRAVE_DIR/History"*
rm -f "$BRAVE_DIR/Shortcuts"
rm -f "$BRAVE_DIR/Top Sites"*
rm -f "$BRAVE_DIR/Network Action Predictor"
```

---

## Quick Reference: Keyboard Shortcuts

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| New Incognito Window | `Cmd + Shift + N` | `Ctrl + Shift + N` |
| Switch Profile | `Cmd + Shift + M` | `Ctrl + Shift + M` |
| Clear Browsing Data | `Cmd + Shift + Delete` | `Ctrl + Shift + Delete` |
| Toggle Bookmarks Bar | `Cmd + Shift + B` | `Ctrl + Shift + B` |
| Open Settings | `Cmd + ,` | `Ctrl + ,` |

---

## Pre-Presentation Checklist

Before going live:

1. **Switch to Teaching profile**: `Cmd + Shift + M`
2. **Hide bookmarks bar**: `Cmd + Shift + B`
3. **Open Incognito** for any new browsing: `Cmd + Shift + N`
4. **Test the URL bar**: Type a few characters - should show no suggestions
5. **Close personal tabs**: Check for any leftover tabs from other sessions

---

## Summary

| Approach | Effort | Effectiveness |
|----------|--------|---------------|
| Separate profile | Low | High |
| Disable suggestions (GUI) | Low | Medium |
| Python script | Medium | High |
| Always use Incognito | None | Medium |

The **dedicated Teaching profile + disabled suggestions** combo gives you the cleanest experience. The Python script makes it reproducible across machines.

---

## Files

The scripts from this post are available in the [blog repository](https://github.com/nipunbatra/blog):

- [`brave_privacy_patch.json`](terminal-optimization-examples/brave_privacy_patch.json)
- [`apply_brave_privacy.py`](terminal-optimization-examples/apply_brave_privacy.py)
