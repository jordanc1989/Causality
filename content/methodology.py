from dash import html


def psm_intro_copy():
    return [
        html.P([
            html.Strong("Why this tab exists: "),
            "This dataset is a randomised experiment, so propensity score matching is "
            "not the right tool for estimating the email effect, random assignment "
            "already balances the arms. This tab is a ",
            html.Em("methodology demonstration"),
            ": it runs the workflow you would use if the data had been observational "
            "(loyalty-program opt-ins, behaviourally-triggered sends, organic signups), "
            "and shows the results sitting alongside the randomised estimates. Treat the "
            "numbers as a sanity check rather than an independent answer!"
        ], className="mb-2 small"),
        html.P([
            html.Strong("How matching works: "),
            "Each email recipient is paired with the control customer whose pre-treatment "
            "profile (spend history, recency, channel, postcode type, catalogue preference, "
            "and the new-customer flag) is closest on the modelled probability of being "
            "treated. The same control can match more than one recipient. If even the "
            "closest control is too different, the recipient is dropped rather than forced "
            "into a poor pair. Because assignment really was random, almost everyone pairs "
            "cleanly and the matched ATT lands close to the randomised difference-in-means."
        ], className="mb-2 small"),
        html.P([
            html.Strong("How to read the panels: "),
            "The Love plot is the diagnostic — it shows whether the two arms look similar "
            "on each covariate before and after matching. In an observational study a big "
            "shift toward zero after matching is the win condition. Here, the ",
            html.Em("before"),
            " plot already shows balance because randomisation did the work; matching "
            "barely changes anything. The bar chart reports the matched ATT with a "
            "stress-test band from redoing the propensity fit and the matching 200 times "
            "on resampled data."
        ], className="mb-0 small text-muted"),
    ]
