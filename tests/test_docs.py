"""The generated documents (README findings, research note) contain no number
that is not read from the committed results, and match what the results produce today."""

from __future__ import annotations

import pytest

from wsl.report.docs import (
    END,
    LEARNED_END,
    LEARNED_START,
    README,
    START,
    TLDR_END,
    TLDR_START,
    TOP_END,
    TOP_START,
    render_all,
    unregistered_numbers,
)


@pytest.fixture(scope="module")
def rendered() -> dict:  # type: ignore[type-arg]
    out = render_all()
    return {"docs": out, "fmts": render_all.fmts}  # type: ignore[attr-defined]


def _block(text: str, start: str = START, end: str = END) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_committed_documents_match_the_results(rendered: dict) -> None:  # type: ignore[type-arg]
    for path, text in rendered["docs"].items():
        assert path.read_text() == text, (
            f"{path.name} differs from what the results produce; run `uv run wsl report docs`")


def test_no_number_is_typed_by_hand(rendered: dict) -> None:  # type: ignore[type-arg]
    fmts = rendered["fmts"]
    for path, text in rendered["docs"].items():
        if path == README:
            bad = unregistered_numbers(_block(text), fmts["readme"])
            bad += unregistered_numbers(_block(text, TLDR_START, TLDR_END), fmts["tldr"])
            bad += unregistered_numbers(_block(text, LEARNED_START, LEARNED_END), fmts["learned"])
            bad += unregistered_numbers(_block(text, TOP_START, TOP_END), fmts["top"])
        else:
            bad = unregistered_numbers(text, fmts["note"])
        assert bad == [], f"{path.name}: numbers not read from the results: {bad}"


def test_the_check_catches_an_invented_number(rendered: dict) -> None:  # type: ignore[type-arg]
    note = next(t for p, t in rendered["docs"].items() if p.name == "RESEARCH_NOTE.md")
    tampered = note.replace("## 1. The question",
                            "Stocks fell 3.7% that day.\n\n## 1. The question")
    assert "3.7" in unregistered_numbers(tampered, rendered["fmts"]["note"])


def test_readme_images_are_files() -> None:
    """Every local image the README points at is in the tree, including the rasters."""
    import re

    from wsl.paths import ROOT

    text = README.read_text()
    paths = re.findall(r'(?:src="|\]\()((?:docs/)[^)"\s]+\.(?:png|gif|svg))', text)
    assert paths
    missing = [p for p in paths if not (ROOT / p).is_file()]
    assert missing == []


def test_readme_top_states_the_live_status(rendered: dict) -> None:  # type: ignore[type-arg]
    from wsl.report.facts import facts

    top = _block(rendered["docs"][README], TOP_START, TOP_END)
    if not facts()["live"]["scores"]:
        assert "no track record" in top
