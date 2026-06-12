"""Numeric formatters used across layouts and callbacks.

Centralising here so spend, percentages and percentage-points are written
consistently everywhere (precision, sign, thousands separator).
"""


def money(x, precision=2):
    """`12345.6 -> '$12,345.60'`, sign-preserving on negatives."""
    if x is None:
        return ""
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{precision}f}"


def money_range(lo, hi, precision=2):
    """`(0.47, 1.03) -> '$0.47–$1.03'`. En-dashed interval, sign-preserving."""
    if lo is None or hi is None:
        return ""
    return f"{money(lo, precision)}–{money(hi, precision)}"


def money_compact(x):
    """Short-form spend used in chart annotations / tight tiles.

    `$1.2k`, `$45k`, `$3.1M`. Falls back to `money(..., 0)` below $1,000.
    """
    if x is None:
        return ""
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1_000_000:
        return f"{sign}${a / 1_000_000:.1f}M"
    if a >= 10_000:
        return f"{sign}${a / 1_000:.0f}k"
    if a >= 1_000:
        return f"{sign}${a / 1_000:.1f}k"
    return f"{sign}${a:,.0f}"


def pct(x, precision=1, signed=False):
    """`0.124 -> '12.4%'`. `x` is a fraction in [0, 1] (or above)."""
    if x is None:
        return ""
    fmt = f"{{:+.{precision}f}}%" if signed else f"{{:.{precision}f}}%"
    return fmt.format(x * 100)


def pp(x, precision=2):
    """`0.34 -> '+0.34pp'`. `x` is in percentage points already."""
    if x is None:
        return ""
    return f"{x:+.{precision}f}pp"
