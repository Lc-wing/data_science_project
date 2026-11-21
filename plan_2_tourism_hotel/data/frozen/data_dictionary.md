# Frozen data dictionary (Plan 2)

`panel_tier_month.csv` has one row per star class and month (804 rows) and carries every column below. `months.csv` has one row per month (201 rows) and carries the month-level columns. Shares are in percent. Derived values are rounded to 10 decimals.

| Column | Level | Unit | Definition |
| --- | --- | --- | --- |
| `tier` | tier-month |  | Star class: 5star, 4star, 3star, 2star (DSEC classes as published) |
| `period` | month | YYYY-MM | Reference month |
| `occupancy_pct` | tier-month | % | Average occupancy rate of guest rooms (DSEC) |
| `rooms` | tier-month | rooms | Guest rooms of operating establishments at month end (DSEC) |
| `log_rooms` | tier-month | log rooms | Natural log of rooms |
| `establishments` | tier-month | count | Operating establishments at month end (DSEC) |
| `guests` | tier-month | persons | Guests of the star class during the month (DSEC) |
| `year` | month |  | Calendar year |
| `month` | month | 1-12 | Calendar month |
| `days_in_month` | month | days | Days in the month |
| `period_label` | month |  | pre_pandemic (2010-01 to 2019-12), transition_2020_01, pandemic (2020-02 to 2022-12), post_pandemic (2023-01 on) |
| `post` | month | 0/1 | 1 in the post-pandemic period |
| `in_main_sample` | month | 0/1 | 1 in the pre-pandemic and post-pandemic periods |
| `visitors_total` | month | persons | Visitor arrivals (DSEC) |
| `visitors_overnight` | month | persons | Overnight visitor arrivals (DSEC) |
| `visitors_same_day` | month | persons | Same-day visitor arrivals (DSEC) |
| `visitors_mainland` | month | persons | Arrivals with a Chinese mainland travel document |
| `visitors_hong_kong` | month | persons | Arrivals with a Hong Kong SAR travel document |
| `visitors_taiwan` | month | persons | Arrivals with a Taiwan region travel document |
| `package_tour_visitors` | month | persons | Visitors on package tours (DSEC) |
| `log_visitors_overnight` | month | log persons | Natural log of visitors_overnight |
| `same_day_share_pct` | month | % | 100 × visitors_same_day / visitors_total |
| `international_share_pct` | month | % | 100 × (1 − (mainland + Hong Kong + Taiwan) / visitors_total) |
| `package_tour_share_pct` | month | % | 100 × package_tour_visitors / visitors_total |
| `cn_holiday_days` | month | days | Days inside a mainland public-holiday period, final schedule (State Council notices). Used in the ex-post regression |
| `cn_golden_week_days` | month | days | Days inside a Spring Festival or National Day holiday period |
| `cn_holiday_days_at_origin` | month | days | cn_holiday_days as scheduled by the notices published up to the end of the month before. This, not cn_holiday_days, is the known calendar feature for forecasts |
| `cn_golden_week_days_at_origin` | month | days | cn_golden_week_days as scheduled by the end of the month before |
| `typhoon_days` | month | days | Days with tropical cyclone signal No. 8 or higher in force (SMG). Known only after the month: used in the ex-post regression and as a lagged forecast feature, never for the target month |

Other files:

- `holiday_periods.csv`: the 115 mainland holiday periods with start, end, and the day the schedule was published, plus the 2 schedules a later notice replaced (`superseded`).
- `rolling_origins.csv`: validation and test origins. The origin is the last month whose data may be used; the target is the next month.
- `split_metadata.json`: the split rules in words and counts.
- `analysis_spec.json`: the pre-registered analysis specification.
- `FREEZE_MANIFEST.json`: SHA-256 of every file above.
