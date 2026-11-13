#!/usr/bin/env python3
"""Check exact publication payloads before sending them."""
import argparse
import getpass
import os
from pathlib import Path
import re
import subprocess
import sys

BLOCKED = "Blocked: local machine identity was found in public text."


def private_terms():
    values = [str(Path.home()), getpass.getuser(), os.getenv("USER", ""), os.getenv("LOGNAME", "")]
    values.extend(os.getenv("WEBSHART_PRIVATE_TERMS", "").splitlines())
    return {value.casefold() for value in values if value and len(value) >= 3}


def contains_private_text(text, terms=None):
    folded = text.casefold()
    if any(term in folded for term in (private_terms() if terms is None else terms)):
        return True
    return bool(re.search(r"(?i)(?:/(?:users|home)/[^\s/]+|[a-z]:[\\/]users[\\/][^\s\\/]+|^\s*(?:co-authored-by|reviewed-by|signed-off-by):)", text, re.MULTILINE))


def git(*args):
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode("utf-8", errors="replace")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--git-range")
    args = parser.parse_args(argv)
    try:
        payloads = [Path(name).read_text(encoding="utf-8") for name in args.text_file]
        if args.staged:
            payloads.append(git("diff", "--cached", "--binary", "--no-ext-diff"))
        if args.git_range:
            payloads.extend([git("diff", "--binary", "--no-ext-diff", args.git_range),
                             git("log", "--format=%B", args.git_range)])
        if not payloads:
            parser.error("Provide an exact text file, staged diff, or commit range")
        if any(contains_private_text(value) for value in payloads):
            print(BLOCKED, file=sys.stderr)
            return 1
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        print("Blocked: public text could not be checked.", file=sys.stderr)
        return 2
    print("Public text check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
