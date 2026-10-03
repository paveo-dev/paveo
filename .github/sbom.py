"""Write the release's software bill of materials, CycloneDX 1.5 JSON.

Paveo installs nothing else, so the bill is short and that is the point: one
component, the package, with the hash of every file published, and the one
optional dependency named with the extra that brings it in (THREAT_MODEL §5).

    python .github/sbom.py dist/ sbom/paveo.cdx.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

# The repository's, wherever this is run from. No timestamp is written, so one
# build always gives the same bill (Rule 14).
_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def main(dist: Path, out: Path) -> None:
    project = tomllib.loads(_PYPROJECT.read_text("utf-8"))["project"]
    name, version = project["name"], project["version"]
    files = sorted(dist.iterdir())
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{name}@{version}",
                "licenses": [{"license": {"id": project["license"]}}],
            },
        },
        "components": [
            {
                "type": "file",
                "name": f.name,
                "hashes": [
                    {
                        "alg": "SHA-256",
                        "content": hashlib.sha256(f.read_bytes()).hexdigest(),
                    }
                ],
            }
            for f in files
        ]
        + [
            {
                "type": "library",
                "name": _name(requirement),
                "scope": "optional",
                "properties": [
                    {"name": "extra", "value": extra},
                    {"name": "requirement", "value": requirement},
                ],
            }
            for extra, requirements in project["optional-dependencies"].items()
            if extra != "dev"
            for requirement in requirements
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")


def _name(requirement: str) -> str:
    """``cryptography>=45,<51`` names ``cryptography``."""
    found = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if found is None:
        raise SystemExit(f"cannot read the requirement {requirement!r}")
    return found.group(0)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
