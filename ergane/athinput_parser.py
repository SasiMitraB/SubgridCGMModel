"""
ergane.athinput_parser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parser for Athena/AthenaK input files (the "athinput" format).

The format is an INI-like file where sections are delimited by angle-bracket
headers (<section>) and each parameter is a ``key = value  # optional comment``
line.  Comment-only lines start with ``#``.

Usage
-----
>>> from ergane.athinput_parser import parse_athinput
>>> params = parse_athinput("kh2d/kh2d-sin.athinput")
>>> params["mesh"]["nx1"]
'256'
>>> params["time"]["tlim"]
'6.0'
"""

from __future__ import annotations

import re
from pathlib import Path


# ── Public API ────────────────────────────────────────────────────────────────

def parse_athinput(path: str | Path) -> dict[str, dict[str, str]]:
    """
    Parse an Athena input file and return a nested dict.

    Parameters
    ----------
    path : str or Path
        Path to the athinput file.

    Returns
    -------
    dict[str, dict[str, str]]
        Outer key = section name (e.g. ``"mesh"``, ``"time"``).
        Inner key = parameter name.
        Values are raw strings — cast them yourself if needed.

    Notes
    -----
    * Section names are lower-cased.
    * Keys and values are stripped of surrounding whitespace.
    * Inline ``# comments`` are stripped from values.
    * Lines that do not contain ``=`` (and are not section headers or pure
      comments) are stored as ``section["__raw__"]`` entries for debugging.
    """
    params: dict[str, dict[str, str]] = {}
    current_section: str | None = None

    section_re = re.compile(r"^<(\w+)>")
    kv_re = re.compile(r"^([^#=\s][^=]*)=([^#]*)")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip blank lines and pure comment lines
            if not line or line.startswith("#"):
                continue

            # Section header
            m = section_re.match(line)
            if m:
                current_section = m.group(1).lower()
                params.setdefault(current_section, {})
                continue

            if current_section is None:
                continue

            # Key = value pair
            m = kv_re.match(line)
            if m:
                key = m.group(1).strip()
                # Strip trailing inline comment from value
                value = m.group(2).split("#")[0].strip()
                params[current_section][key] = value

    return params


def typed(params: dict[str, dict[str, str]], section: str, key: str, cast=str):
    """
    Convenience helper — retrieve and cast a parameter value.

    >>> typed(params, "mesh", "nx1", int)
    256
    """
    return cast(params[section][key])
