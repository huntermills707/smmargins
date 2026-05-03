"""Colorblind-safe default palette."""

_DEFAULT_COLORS = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#D55E00",  # red
    "#CC78BC",  # purple
    "#CA9161",  # brown
    "#FBAFE4",  # pink
    "#949494",  # gray
    "#56B4E9",  # light blue
    "#00D2D5",  # cyan
]


def get_color(i: int) -> str:
    return _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
