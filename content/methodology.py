from dash import html


def psm_intro_copy():
    return [
        html.P([
            html.Strong("What this does: "),
            "Each customer who got an email is paired with a control customer who looked "
            "almost identical beforehand (same purchase history, recency, channel, and so on). "
            "Comparing matched pairs strips out any pre-existing differences between the two "
            "groups and isolates the effect of the email itself."
        ], className="mb-2 small"),
        html.P([
            html.Strong("How customers are matched: "),
            "For each email recipient we find the closest lookalike in the control group on "
            "things visible before the send: spend history, recency, channel, postcode type, "
            "catalogue preference, and a few flags. The same control customer can be the "
            "nearest match for more than one recipient. If even the closest control looks too "
            "different, we drop that recipient from this view rather than stretch the comparison. "
            "Because assignment was random to begin with, almost everyone pairs cleanly and the "
            "answer barely moves. The full algebra (propensity model, caliper, bootstrap) is in "
            "the Methodology section below."
        ], className="mb-2 small"),
        html.P([
            html.Strong("Why this matters: "),
            "This tab is laid out like the ", html.Em("observational"),
            " dashboards marketers fall back on when there's no random assignment to lean on. "
            "It doesn't replace the trustworthy numbers on the Overview or the OLS tab, it sits "
            "alongside them as a sanity check. The Love plot shows whether the two groups look "
            "similar before and after matching. The bar chart repeats the lift with a confidence "
            "band built by redoing the matching 200 times on resampled data."
        ], className="mb-0 small text-muted"),
    ]
