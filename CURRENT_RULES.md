# Current Rules

Living checklist of the currently-active spec, kept up to date as the client
changes his mind. When a rule changes, update this file in the same commit
as the code change. The code comments cite dates/quotes for history; this
file states only what is true RIGHT NOW, so it can be checked against
without re-reading the whole codebase or chat history.

Last updated: 2026-09-13.

## Market Entry window

- **9:32am to 10:30am ET** (`EARLIEST_ENTRY_TIME` / `LATEST_ENTRY_TIME`,
  `phase2/risk_manager.py`). 9:30 → 9:31 (2026-09-12) → 9:32 (2026-09-13).

## Expansion Leg (which stock, which direction)

- Baseline is the **Rolling Equilibrium Price**: each leg's own mean price
  over `price_history_a/b` (yesterday's 390 RTH bars + today's RTH so
  far), not a literal Kalman filter (a simple mean, mirroring
  `equilibrium_spread`'s own "rolling mean, not Kalman" precedent —
  flagged as an assumption). Replaced the 2026-09-03 "current divergence
  event's own start price" baseline 2026-09-13.
- Dominant leg must be responsible for **≥70%** of the divergence
  expansion (`DRIVER_STOCK_MIN_SHARE`, `phase2/main.py`) — a close split
  (e.g. 55/45) no longer trades at all, even though one leg is still
  nominally larger.

## Pair selection tiebreak (Adjusted Spread Efficiency)

- When exactly one pair is eligible in a check cycle, it executes directly —
  no ranking, no probing.
- When **2+ pairs are eligible at once**, the one with the highest
  **Spread Return % = (Divergence $ / Stock Initial Margin) x 100** wins
  (`select_highest_adjusted_spread_efficiency`, `phase2/risk_manager.py`).
  Fully replaced the old "closest to trigger" tiebreak 2026-09-12
  (`select_closest_to_trigger` still exists, dormant, unused).
- Stock Initial Margin is real-order-probed **concurrently** for every
  eligible candidate (`rank_eligible_signals_by_adjusted_spread_efficiency`,
  `phase2/main.py`), not just the eventual winner. Three tiers per
  candidate: quantity=1 first, then the same guessed quantity the real
  entry would start from, then the flat `INITIAL_MARGIN_FRACTION` estimate
  as a last resort if neither real probe actually rejects.

## Divergence Watch display

- A pair with a real trigger shows on the Watch **even before live data
  arrives** (e.g. right at 9am from the cache load) — `current`/`distance`
  are `null` until the first live tick, not excluded entirely.
- Each row carries a **`cleared`** flag: true only when BOTH (a)
  `reverted_over_50pct_since_max_divergence` — the pair pulled back more
  than 50% from its own peak divergence at some point after the peak (a
  DISTINCT condition from the dormant 100%-reversion Condition A below,
  not the same thing) AND (b) `not trigger_reached_today` — no retouch of
  the trigger yet this session (lookback = current RTH, not the prior
  session).
- `reverted_over_50pct_since_max_divergence` is ALSO a live **entry gate**
  now (2026-09-13: "it should have at least >50% mean reversion before
  entry") — `entry_eligible` blocks entry until it's True, not just a
  display label.

## Divergence range (MIN_ENTRY_DIVERGENCE / MAX_ENTRY_DIVERGENCE)

- Currently **1.5 to 5**.
- Applies to the **trigger (Maximum Divergence) only** — a trigger outside
  this range is discarded, the pair never gets tracked at all
  (`_bounded_trigger_level`, `phase2/main.py`).
- Does **not** apply to today's live current value. Current is governed
  entirely by the Entry Window below.

## Entry Window

- **13% to 49.9%** of the trigger level (`ENTRY_TRIGGER_DISTANCE_MIN_PCT` /
  `MAX_PCT`, `phase2/risk_manager.py`).
- `distance = trigger - abs(current deviation)`. Eligible when
  `0.13 * trigger <= distance <= 0.499 * trigger`.
- Scales per pair, off that pair's own trigger — not a flat dollar range
  applied the same way to every pair.

## Johansen / Hurst (20-day)

- Live, continuously rechecked gate. Sticky once cleared this session —
  clearing once is enough, they don't need to clear together.
- `MIN_HURST = 0.40`.
- **Pairs loaded from the prior RTH session (the 9am cache)** are
  pre-cleared at warm-up — they skip the live re-validation entirely and go
  straight to Divergence Watch once other conditions are met
  (`_apply_warmup_data`, `phase2/main.py`).
- **Pairs found via the live mid-session universe scan**
  (`run_universe_screen_in_background`) still validate Johansen/Hurst live,
  same as before — the pre-clear above does not apply to them.
- `trigger_reached_today` is separately sticky: once a pair's current
  deviation has reached or crossed its trigger at any point in the
  session, it's disqualified for the rest of the day even if it pulls back
  into the Entry Window again later.

## 9am cache (tomorrow's qualified pairs list)

- Built once, automatically, at the end of each session
  (`cache_qualified_pairs_for_tomorrow`) by
  `select_pairs_from_full_universe_prior_session`.
- **Full S&P 500 scan** of the prior RTH session (same-industry
  combinations, `universe_screening.candidate_pairs`) — not scoped to the
  Market Entry List.
- Qualifies on Pearson 60-day correlation ≥ 0.90 AND Pearson 20-day
  correlation < 0.70 (the breakdown/expansion test), same as always.
- **Also** requires Johansen and Hurst to have cleared at some point during
  the prior session (9:30am–4pm), replayed bar-by-bar
  (`_johansen_hurst_ever_cleared_in_prior_session`) — not a single
  whole-window snapshot.
- The **live 9:30am fallback**, `select_pairs_from_previous_session`, is
  unchanged and still Market-Entry-List-scoped, Pearson-only (no
  Johansen/Hurst check). It only runs if the 9am cache is missing/stale,
  and stays fast on purpose since it can't delay the entry window.

## Check cycle timing

- Fires just after each real 1-minute bar closes
  (`_seconds_until_next_bar_close`), not a flat 60-second sleep from
  whenever the previous cycle finished.

## Entry sizing / margin

- Real-order margin probe: submits the real candidate entry plus all 4
  real bracket legs at deliberately un-triggerable prices, reads the
  required margin off IBKR's own 201 rejection, resizes, up to 6 rounds.
  This is the FINAL sizing probe for whichever pair actually gets chosen.
- `INITIAL_MARGIN_FRACTION` (0.35) is only a starting-quantity estimate
  before the first probe, not the final sizing. Also used as the last-tier
  fallback in the Adjusted Spread Efficiency ranking probe above.
- The ranking probe (above) is a SEPARATE, earlier set of real probes run
  per candidate before one is chosen — not the same probe as this one,
  which only runs once, for the winner.

## Stop losses

- **Hard Stop Loss** — not a flat percentage. Placed at entry via
  `wide_hard_stop_price` (Trigger reduced by the entry window's own 13%
  floor, plus Trigger itself), then re-set moments later via
  `resting_hard_stop_price` using the real entry divergence
  (`entry_divergence + trigger`, "100% Mean Reversion x 2").
- **PnL Stop Loss** — `MAX_LOSS_PCT = 0.05`, 5% of EquityWithLoanValue.
(Client raised lowering to 2.5% on 2026-09-09, then said "Hold on Stop
Loss" the same day — reverted, both stayed at their prior values above.)

## Disabled (dormant, not deleted)

- News Stop (exit) — `NEWS_STOP_ENABLED = False`.
- Excluded List — same flag.
- 1-day Johansen / Hurst / Pearson gates — computed and shown on the
  dashboard's raw Live Validation panel, no longer gate anything.
- Condition A (100% mean reversion after Maximum Divergence,
  `mean_reverted_since_max_divergence`) — computed, no longer gates entry.
  Distinct from the ACTIVE `reverted_over_50pct_since_max_divergence`
  (>50%, not 100%) that feeds the Divergence Watch's `cleared` flag above
  — the client has explicitly said the old 100% one "does not exist" as a
  condition, don't conflate the two.
- Two-leg / hedge execution — built, not wired in, single-leg only.
