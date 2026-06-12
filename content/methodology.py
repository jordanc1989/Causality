from dash import html


def psm_intro_copy():
    return [
        html.P([
            "This is a randomised experiment, so matching isn't the right tool here. "
            "Random assignment already balances the groups. The tab runs the workflow you'd "
            "use on ",
            html.Em("observational"),
            " data (opt-ins, triggered sends, organic signups) and sets the result next to "
            "the randomised estimates. Read it as a sanity check, not an independent answer."
        ], className="mb-2 small"),
        html.P([
            "Each recipient is paired with the control customer whose pre-treatment profile "
            "is closest on the modelled probability of being treated - a recipient with no "
            "close match is dropped rather than forced into a poor pair. Because assignment "
            "really was random, almost everyone pairs cleanly and the matched ATT lands "
            "close to the simple difference-in-means."
        ], className="mb-2 small"),
        html.P([
            "The Love plot shows whether the groups look alike on each covariate, before and "
            "after matching. Here "
            "the ",
            html.Em("before"),
            " plot is already balanced, so matching barely moves it. The ATT chart shows "
            "the matched-pair interval beside a stress-test band from refitting and "
            "rematching 200 times."
        ], className="mb-0 small text-muted"),
    ]
