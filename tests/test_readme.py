"""Tests for README.md packaging correctness.

README.md is shipped verbatim as the PyPI ``long_description`` (see setup.py).
PyPI cannot resolve relative image paths, so every image ``src`` must be an
absolute URL or the logo/images render broken on the project page.

See https://github.com/ZoneMinder/pyzmNg/issues/49
"""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

# Matches src="..." in <img> tags and ![alt](...) markdown images.
_HTML_IMG_SRC = re.compile(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
_MD_IMG_SRC = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")


def _image_srcs() -> list[str]:
    text = README.read_text(encoding="utf-8")
    return _HTML_IMG_SRC.findall(text) + _MD_IMG_SRC.findall(text)


def test_readme_exists():
    assert README.is_file()


def test_all_readme_images_are_absolute_urls():
    """Relative image paths render broken on PyPI; require absolute URLs."""
    srcs = _image_srcs()
    assert srcs, "expected at least one image in README.md"
    relative = [s for s in srcs if not s.startswith(("http://", "https://"))]
    assert not relative, (
        f"README image src(s) must be absolute URLs for PyPI: {relative}"
    )
