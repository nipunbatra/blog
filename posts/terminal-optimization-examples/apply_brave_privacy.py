#!/usr/bin/env python3
"""
Apply privacy settings patch to Brave Browser.
Only modifies specified fields, preserves rest of config.

Usage:
    python apply_brave_privacy.py [--patch patch.json] [--dry-run]
"""

import json
import platform
from pathlib import Path
import shutil
from datetime import datetime


def get_brave_prefs_path() -> Path:
    """Get Brave preferences path for current OS."""
    system = platform.system()
    home = Path.home()

    paths = {
        "Darwin": home / "Library/Application Support/BraveSoftware/Brave-Browser/Default/Preferences",
        "Linux": home / ".config/BraveSoftware/Brave-Browser/Default/Preferences",
        "Windows": home / "AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/Preferences",
    }
    return paths.get(system)


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge patch into base dict."""
    result = base.copy()
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_patch(prefs_path: Path, patch: dict, dry_run: bool = False) -> list:
    """Apply patch to preferences file. Returns list of changes made."""

    with open(prefs_path, 'r') as f:
        original = f.read()
        prefs = json.loads(original)

    # Track changes
    changes = []

    def record_changes(patch_dict, prefix=""):
        for key, value in patch_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                record_changes(value, full_key)
            else:
                changes.append(f"{full_key} = {value}")

    record_changes(patch)

    # Merge
    merged = deep_merge(prefs, patch)

    if not dry_run:
        # Backup
        backup = prefs_path.with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy(prefs_path, backup)

        # Write with same formatting style as original (detect indent)
        with open(prefs_path, 'w') as f:
            json.dump(merged, f, indent=3)  # Brave uses indent=3

    return changes


# Default patch for teaching/presentation privacy
DEFAULT_PATCH = {
    "search": {
        "suggest_enabled": False
    },
    "ntp": {
        "num_personal_suggestions": 0
    },
    "omnibox": {
        "local_history_zero_suggest_eager_loading_enabled": False,
        "rich_autocompletion_full_url_enabled": False
    },
    "documentsuggest": {
        "enabled": False
    },
    "dns_prefetching": {
        "enabled": False
    },
    "network_prediction_options": 2,
    "brave": {
        "autocomplete_enabled": False,
        "top_site_suggestions": False
    }
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply privacy patch to Brave")
    parser.add_argument("--patch", help="Custom patch JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--show-patch", action="store_true", help="Print default patch and exit")
    args = parser.parse_args()

    if args.show_patch:
        print(json.dumps(DEFAULT_PATCH, indent=2))
        exit(0)

    prefs_path = get_brave_prefs_path()
    if not prefs_path or not prefs_path.exists():
        print("Error: Brave preferences not found")
        exit(1)

    # Load patch
    if args.patch:
        with open(args.patch) as f:
            patch = json.load(f)
    else:
        patch = DEFAULT_PATCH

    print("Brave Privacy Patch")
    print("=" * 40)
    print(f"Preferences: {prefs_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print()

    changes = apply_patch(prefs_path, patch, dry_run=args.dry_run)

    print("Changes:")
    for c in changes:
        print(f"  {c}")

    if not args.dry_run:
        print("\n✓ Applied! Restart Brave.")
    else:
        print("\n(dry run - no changes made)")
