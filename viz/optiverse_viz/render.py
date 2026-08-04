"""The view model, as one HTML file.

Everything is inlined — the vendored d3 build, the page's own script and style,
and the run's data as JSON. The file has no second half it can be separated
from: it opens over `file://`, it opens with the network off, and it can be
copied out of the run and mailed somewhere and still be the same page. A report
that only renders next to the assets it was built beside is a report nobody can
send.

The vendored build is d3 7.9.0 (`assets/d3.v7.min.js`, ISC, licence beside it).
Only `d3-hierarchy`, `d3-selection`, `d3-shape` and `d3-zoom` are used, but
cutting a custom bundle needs a node toolchain neither project has, so the whole
build is carried.
"""

import json
import string
from pathlib import Path
from typing import Optional

from .model import View, build

ASSETS = Path(__file__).parent / "assets"

TEMPLATE_NAME = "index.html.tmpl"
D3_NAME = "d3.v7.min.js"
SCRIPT_NAME = "app.js"
STYLE_NAME = "app.css"

# Where the report lands inside the run it describes, so it travels with it. The
# store ignores anything here, so writing it into a run that is still going is
# invisible to the loop.
OUTPUT_DIRECTORY_NAME = "viz"
OUTPUT_NAME = "index.html"


def render(directory: Path) -> str:
    """The page for the run at `directory`."""
    return _fill(build(directory))


def write(directory: Path, output: Optional[Path] = None) -> Path:
    """Render the run at `directory` and return where the page went."""
    destination = (
        Path(output)
        if output is not None
        else Path(directory) / OUTPUT_DIRECTORY_NAME / OUTPUT_NAME
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(Path(directory)))

    return destination


def _fill(view: View) -> str:
    template = string.Template(_asset(TEMPLATE_NAME))

    return template.substitute(
        data=_payload(view),
        d3=_asset(D3_NAME),
        script=_asset(SCRIPT_NAME),
        style=_asset(STYLE_NAME),
        title=Path(view["run"]["path"]).name or view["run"]["path"],
    )


def _payload(view: View) -> str:
    """The data, safe to sit inside a `<script>`.

    A `<` anywhere in the JSON can close the element early — a constraint reading
    "use `</script>` tags" is prose a director could plausibly write — so every
    one of them is escaped. It is still the same JSON to a parser.
    """
    return json.dumps(view, indent=1).replace("<", "\\u003c")


def _asset(name: str) -> str:
    return (ASSETS / name).read_text()
