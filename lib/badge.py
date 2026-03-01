"""
SVG badge generator for MuniMet.ro status.

Generates shields.io-style flat badges showing current system status.
"""

# Badge colors (shields.io conventions)
_COLORS = {
    'green': '#4c1',
    'yellow': '#dfb317',
    'red': '#e05d44',
    'unknown': '#9f9f9f',
}

_LABELS = {
    'green': 'on track',
    'yellow': 'delays',
    'red': 'down',
    'unknown': 'unknown',
}

# Approximate character width for Verdana 11px (shields.io uses this)
_CHAR_WIDTH = 6.8
_PADDING = 10
_LABEL_TEXT = 'MuniMet.ro'


def generate_badge(status):
    """
    Generate an SVG badge for the given status.

    Args:
        status: One of 'green', 'yellow', 'red', or None for unknown.

    Returns:
        str: SVG markup string.
    """
    status = status if status in _COLORS else 'unknown'
    color = _COLORS[status]
    value_text = _LABELS[status]

    # Calculate widths
    label_width = int(len(_LABEL_TEXT) * _CHAR_WIDTH + _PADDING * 2)
    value_width = int(len(value_text) * _CHAR_WIDTH + _PADDING * 2)
    total_width = label_width + value_width

    label_x = label_width / 2
    value_x = label_width + value_width / 2

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{_LABEL_TEXT}: {value_text}">
  <title>{_LABEL_TEXT}: {value_text}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
    <rect width="{total_width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
    <text aria-hidden="true" x="{label_x * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)">{_LABEL_TEXT}</text>
    <text x="{label_x * 10}" y="140" transform="scale(.1)">{_LABEL_TEXT}</text>
    <text aria-hidden="true" x="{value_x * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)">{value_text}</text>
    <text x="{value_x * 10}" y="140" transform="scale(.1)">{value_text}</text>
  </g>
</svg>'''
