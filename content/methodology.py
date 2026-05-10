from dash import html


def psm_intro_copy():
    return [
        html.P([
            html.Strong("What this does: "),
            "Each customer who received an email is paired with a control customer "
            "who looked almost identical beforehand (same purchase history, recency, "
            "channel and demographics). Comparing paired customers isolates the email's "
            "effect from any pre-existing differences between groups."
        ], className="mb-2 small"),
        html.P([
            html.Strong("How customers are matched: "),
            "For each email recipient we search the control group for the closest lookalike ",
            "on things you could see before the send: spend history, recency, channel, ",
            "broad postcode type, which catalogues someone leans toward, and similar ",
            "flags. If even the nearest control still looks too different, ",
            "we omit that shopper from this sensitivity view instead of stretching the comparison. ",
            "Assignments were ", html.Strong("randomised"), ", so almost everyone pairs cleanly ",
            "and the headline uplift barely budges versus always grabbing the nearest neighbour. ",
            "For the algebraic details (propensity model, logit caliper, pooled bootstrap), expand ",
            html.Strong("Methodology"), " below.",
        ], className="mb-2 small"),
        html.P([
            html.Strong("Why this matters: "),
            "This view is styled like the ", html.Em("observational"),
            " dashboards marketers run when randomisation is unavailable; "
            "it does not replace the Overview or Tab 5 estimates that lean on the ", html.Strong("randomised"),
            " assignment. The Love plot shows whether shopper traits line up sensibly ",
            html.Em("before"), " and ", html.Em("after"),
            " pairing, and the bar chart repeats the lift with a band computed by redoing "
            "matching 200 times on resampled data (a practical stress test rather than textbook matching confidence intervals)."
        ], className="mb-0 small text-muted"),
    ]
