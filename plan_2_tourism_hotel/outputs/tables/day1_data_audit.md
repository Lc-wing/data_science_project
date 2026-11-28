# Plan 2 Day-1 data audit

Written by `scripts/01_download_and_audit.py` from the raw snapshots in `data/raw/`. Downloads ran between 2026-11-21T12:25:51+08:00 and 2026-11-21T12:30:01+08:00 (UTC+8); `data/raw/data_provenance.json` lists the URL, time, size, and SHA-256 of all 176 files.

## Panel window

- 2010-01 to 2026-09: 201 months × 4 star classes = 804 tier-months.
- Pre-pandemic 120 months, 2020-01 transition 1, pandemic 35, post-pandemic 45.

## DSEC series (plan items 1 and 2)

- 34 monthly series from the DSEC Time Series Database, the database behind the `id=402` and `id=403` statistics pages. Every indicator id was checked against DSEC's own label, unit, and parent node before use.
- 16 of 16 star-class series (occupancy, guest rooms, operating establishments, guests × 4 classes) have a value in every month of the window: 804 tier-month occupancy observations.
- Core series with gaps in the window: none.
- Same-day + overnight = total visitors in 201 of 201 months; sea + land + air = total in 201 of 201.
- Four-class sum against the DSEC hotel total: rooms exact in 201 of 201 months, establishments in 201 of 201, guests in 201 of 201.
- Latest Excel releases (CPE_EOH_FR_2026_M09.xlsx, CPE_MV_FR_2026_M10.xlsx): 22 of 22 printed values equal the database values.

### Coverage by series

| key | indicator_id | unit | first_month | last_month | panel_months_with_value | panel_months_missing | missing_span | zero_by_remark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupancy_5star | 13011 | % | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| occupancy_4star | 13012 | % | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| occupancy_3star | 13013 | % | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| occupancy_2star | 13014 | % | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| occupancy_hotels | 13016 | % | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| occupancy_guesthouses | 13015 | % | 2008-01 | 2021-12 | 144 | 57 | 2022-01..2026-09 | 0 |
| occupancy_economical | 67592 | % | 2022-01 | 2026-09 | 57 | 144 | 2010-01..2021-12 | 0 |
| rooms_5star | 25008 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| rooms_4star | 25009 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| rooms_3star | 25010 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| rooms_2star | 25011 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| rooms_hotels | 25007 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| establishments_5star | 25002 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| establishments_4star | 25003 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| establishments_3star | 25004 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| establishments_2star | 25005 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| establishments_hotels | 25001 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| guests_5star | 25093 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| guests_4star | 25122 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| guests_3star | 25151 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| guests_2star | 25180 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| guests_hotels | 25064 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 0 |
| package_tour_visitors | 26001 | No. | 2008-01 | 2026-09 | 201 | 0 |  | 4 |
| visitors_total | 14040 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 0 |
| visitors_same_day | 14272 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 0 |
| visitors_overnight | 57251 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 0 |
| visitors_mainland | 14042 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 0 |
| visitors_hong_kong | 14043 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 0 |
| visitors_taiwan | 14044 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 0 |
| visitors_by_sea | 14024 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 5 |
| visitors_by_land | 14028 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 0 |
| visitors_by_air | 14032 | No. | 2008-01 | 2026-10 | 201 | 0 |  | 0 |
| stay_all_visitors | 14302 | Day | 2011-01 | 2026-10 | 189 | 12 | 2010-01..2010-12 | 0 |
| stay_overnight_visitors | 14330 | Day | 2011-01 | 2026-10 | 189 | 12 | 2010-01..2010-12 | 0 |

### Star-class continuity (plan item 2)

| tier | occupancy_months | rooms_months | establishments_pre_pandemic | establishments_pandemic | establishments_post_pandemic | months_establishment_count_changed | largest_monthly_room_change_pct | month_of_largest_room_change | occupancy_min | occupancy_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5star | 201 | 201 | 22-36 | 31-36 | 35-40 | 32 | 14.5 | 2020-03 | 5 | 96.5 |
| 4star | 201 | 201 | 13-17 | 14-18 | 18-20 | 20 | 30.2 | 2012-04 | 16.8 | 94.6 |
| 3star | 201 | 201 | 11-16 | 13-16 | 17-20 | 16 | 116.7 | 2016-09 | 18.4 | 97.2 |
| 2star | 201 | 201 | 11-15 | 14-22 | 22-26 | 19 | 26.4 | 2020-09 | 11.7 | 93.6 |

- The four star classes are published under the same labels in all 201 months: yes.
- The class below 2-star changed its label: `Guest-houses` ends 2021-12, `Economical accommodation establishments` starts 2022-01. It is outside the four classes analysed.
- The number of operating establishments in a class changed in 87 tier-months (`day1_tier_establishment_changes.csv`). In 5 months one class lost establishments while another gained them; the database does not say whether these are re-ratings or unrelated openings and closures.
- Guest rooms of a class moved by 10% or more from one month to the next in 12 tier-months (pre_pandemic), 0 tier-months (transition_2020_01), 8 tier-months (pandemic), 2 tier-months (post_pandemic). The largest is 3star in 2016-09: 2528 to 5477 rooms (+116.7%).
- DSEC note under the star-class tables: "Note: Data exclude hotel establishments that suspended business and those that did not provide accommodation services to the public."
- Footnotes DSEC attaches to star-class values in the database: Revised figures.
- Reading: the class labels are continuous, the membership of each class is not fixed. Guest rooms per class are in the data so that later steps can control for supply; no class is merged, dropped, or re-labelled.

Months with offsetting moves between classes:

| period | moves |
| --- | --- |
| 2020-08 | 4star 15->14 est., rooms 6186->5886; 3star 15->16 est., rooms 4919->4957 |
| 2020-09 | 5star 34->33 est., rooms 21957->21672; 2star 15->16 est., rooms 980->1239 |
| 2021-01 | 5star 33->32 est., rooms 21684->21351; 4star 15->16 est., rooms 6174->6747 |
| 2021-07 | 5star 32->33 est., rooms 21320->21711; 2star 15->14 est., rooms 1193->1095 |
| 2022-02 | 5star 34->33 est., rooms 24340->23950; 4star 17->18 est., rooms 7074->7244; 2star 21->22 est., rooms 1515->1561 |

Months with a room change of 10% or more:

| tier | period | rooms_before | rooms_after | rooms_change_pct | establishments_before | establishments_after |
| --- | --- | --- | --- | --- | --- | --- |
| 5star | 2011-05 | 12041 | 13407 | 11.3 | 23 | 26 |
| 5star | 2012-09 | 14822 | 16626 | 12.2 | 27 | 28 |
| 5star | 2013-02 | 16594 | 18688 | 12.6 | 28 | 28 |
| 5star | 2020-03 | 24539 | 20992 | -14.5 | 36 | 33 |
| 5star | 2021-12 | 21870 | 24521 | 12.1 | 33 | 34 |
| 5star | 2022-07 | 22032 | 19716 | -10.5 | 33 | 31 |
| 5star | 2022-08 | 19716 | 22051 | 11.8 | 31 | 33 |
| 5star | 2023-04 | 22767 | 25870 | 13.6 | 35 | 35 |
| 4star | 2012-04 | 4269 | 5557 | 30.2 | 13 | 14 |
| 4star | 2015-10 | 6120 | 7718 | 26.1 | 16 | 17 |
| 3star | 2016-09 | 2528 | 5477 | 116.7 | 13 | 15 |
| 3star | 2022-07 | 5226 | 2613 | -50 | 16 | 13 |
| 3star | 2022-08 | 2613 | 5154 | 97.2 | 13 | 15 |
| 2star | 2015-01 | 815 | 655 | -19.6 | 12 | 11 |
| 2star | 2015-03 | 655 | 815 | 24.4 | 11 | 12 |
| 2star | 2016-05 | 815 | 654 | -19.8 | 12 | 11 |
| 2star | 2017-11 | 654 | 791 | 20.9 | 11 | 12 |
| 2star | 2018-01 | 780 | 873 | 11.9 | 12 | 14 |
| 2star | 2019-12 | 873 | 970 | 11.1 | 14 | 15 |
| 2star | 2020-09 | 980 | 1239 | 26.4 | 15 | 16 |
| 2star | 2022-01 | 1200 | 1515 | 26.2 | 15 | 21 |
| 2star | 2023-08 | 1626 | 1812 | 11.4 | 23 | 24 |

### DSEC footnotes in the window

| key | remark | months | first_month | last_month |
| --- | --- | --- | --- | --- |
| guests_2star | Revised figures | 1 | 2026-08 | 2026-08 |
| guests_3star | Revised figures | 1 | 2026-08 | 2026-08 |
| guests_5star | Revised figures | 1 | 2026-08 | 2026-08 |
| guests_hotels | Revised figures | 1 | 2026-08 | 2026-08 |
| occupancy_economical | No figure provided | 144 | 2010-01 | 2021-12 |
| occupancy_guesthouses | No figure provided | 57 | 2022-01 | 2026-09 |
| package_tour_visitors | Absolute value equals zero | 4 | 2020-04 | 2020-07 |
| package_tour_visitors | Revised figures | 3 | 2026-03 | 2026-08 |
| visitors_by_sea | Absolute value equals zero | 5 | 2020-04 | 2022-07 |

### Latest release against the database

| key | period | xlsx_value | xlsx_scale | tsdb_value | match |
| --- | --- | --- | --- | --- | --- |
| establishments_5star | 2026-09 | 40 | 1 | 40 | True |
| rooms_5star | 2026-09 | 26.6 | 1000 | 26578 | True |
| occupancy_5star | 2026-09 | 90.1 | 1 | 90.1 | True |
| guests_5star | 2026-09 | 590.5 | 1000 | 590547 | True |
| establishments_4star | 2026-09 | 20 | 1 | 20 | True |
| rooms_4star | 2026-09 | 9.2 | 1000 | 9233 | True |
| occupancy_4star | 2026-09 | 87 | 1 | 87 | True |
| guests_4star | 2026-09 | 258.8 | 1000 | 258768 | True |
| establishments_3star | 2026-09 | 20 | 1 | 20 | True |
| rooms_3star | 2026-09 | 6.9 | 1000 | 6878 | True |
| occupancy_3star | 2026-09 | 83.7 | 1 | 83.7 | True |
| guests_3star | 2026-09 | 205.6 | 1000 | 205574 | True |
| establishments_2star | 2026-09 | 26 | 1 | 26 | True |
| rooms_2star | 2026-09 | 2 | 1000 | 2021 | True |
| occupancy_2star | 2026-09 | 85.4 | 1 | 85.4 | True |
| guests_2star | 2026-09 | 70.5 | 1000 | 70525 | True |
| visitors_total | 2026-10 | 3823243 | 1 | 3823243 | True |
| visitors_mainland | 2026-10 | 2833998 | 1 | 2833998 | True |
| visitors_hong_kong | 2026-10 | 644741 | 1 | 644741 | True |
| visitors_taiwan | 2026-10 | 94790 | 1 | 94790 | True |
| visitors_same_day | 2026-10 | 2414299 | 1 | 2414299 | True |
| visitors_overnight | 2026-10 | 1408944 | 1 | 1408944 | True |

## Mainland public holidays (plan item 4)

- 17 State Council notices (2010–2026) and 3 later notices that changed a schedule (September 2015, May 2019, Spring Festival 2020).
- 115 holiday periods. 94 of the 94 periods whose notice states a length parse to exactly that length.
- Period rule: the days a notice declares off, joined with adjoining Saturdays and Sundays that the notice does not declare as working days.
- 492 holiday days in the window (243 in Spring Festival or National Day periods), in 117 months.

## SMG tropical cyclone signals (plan item 4)

- 104 SMG tropical cyclone reports, 367 signal records from 2010-07-14T22:00 to 2026-09-03T21:00.
- Typhoon day: calendar day (Macao time) on which signal No. 8 or higher was in force for any length of time.
- 44 typhoon days in the window, in 26 months. By year: 2010: 0, 2011: 1, 2012: 4, 2013: 1, 2014: 1, 2015: 0, 2016: 1, 2017: 3, 2018: 4, 2019: 2, 2020: 3, 2021: 4, 2022: 7, 2023: 5, 2024: 2, 2025: 6, 2026: 0.
- SMG's annual signal table agrees with the parsed reports in 16 of 16 years (number of cyclones reaching signals 1, 3, 8, 9, and 10; dates on which the first and the last cyclone of the year got their first signal).
- Not yet in the annual table: 2026. For its months in the window, 9 of 9 SMG monthly climate summaries list the same signal records as the reports, and the summaries of months without a report have no tropical cyclone section.

### Annual table against parsed reports

| year | in_annual_table | parsed_signal1 | smg_signal1 | parsed_signal3 | smg_signal3 | parsed_signal8 | smg_signal8 | parsed_signal9 | smg_signal9 | parsed_signal10 | smg_signal10 | parsed_first | parsed_last | smg_first | smg_last | match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2010 | True | 5 | 5 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 07/14 | 10/19 | 07/14 | 10/19 | True |
| 2011 | True | 5 | 5 | 4 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 06/10 | 10/02 | 06/10 | 10/02 | True |
| 2012 | True | 5 | 5 | 3 | 3 | 3 | 3 | 1 | 1 | 0 | 0 | 06/17 | 08/24 | 06/17 | 08/24 | True |
| 2013 | True | 7 | 7 | 5 | 5 | 1 | 1 | 0 | 0 | 0 | 0 | 06/21 | 11/01 | 06/21 | 11/01 | True |
| 2014 | True | 3 | 3 | 2 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 07/17 | 09/15 | 07/17 | 09/15 | True |
| 2015 | True | 3 | 3 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 06/21 | 10/02 | 06/21 | 10/02 | True |
| 2016 | True | 8 | 8 | 4 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 05/26 | 10/20 | 05/26 | 10/20 | True |
| 2017 | True | 8 | 8 | 4 | 4 | 3 | 3 | 1 | 1 | 1 | 1 | 06/11 | 10/13 | 06/11 | 10/13 | True |
| 2018 | True | 7 | 7 | 5 | 5 | 2 | 2 | 1 | 1 | 1 | 1 | 06/05 | 10/31 | 06/05 | 10/31 | True |
| 2019 | True | 5 | 5 | 3 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 07/01 | 09/01 | 07/01 | 09/01 | True |
| 2020 | True | 4 | 4 | 5 | 5 | 2 | 2 | 1 | 1 | 1 | 1 | 06/12 | 10/22 | 06/12 | 10/22 | True |
| 2021 | True | 8 | 8 | 4 | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 06/11 | 12/20 | 06/11 | 12/20 | True |
| 2022 | True | 6 | 6 | 5 | 5 | 4 | 4 | 0 | 0 | 0 | 0 | 06/29 | 10/31 | 06/29 | 10/31 | True |
| 2023 | True | 5 | 5 | 3 | 3 | 3 | 3 | 1 | 1 | 1 | 1 | 07/14 | 10/05 | 07/14 | 10/05 | True |
| 2024 | True | 7 | 7 | 5 | 5 | 1 | 1 | 0 | 0 | 0 | 0 | 05/30 | 11/18 | 05/30 | 11/18 | True |
| 2025 | True | 13 | 13 | 8 | 8 | 4 | 4 | 2 | 2 | 2 | 2 | 06/11 | 11/10 | 06/11 | 11/10 | True |
| 2026 | False | 3 |  | 2 |  | 0 |  | 0 |  | 0 |  | 07/01 | 08/31 |  |  |  |

### Monthly climate summaries against parsed reports

| period | summary_file | is_summary_of_month | summary_mentions_cyclone | summary_records | report_records | match |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-01 | c_resumo_202601.pdf | True | False | 0 | 0 | True |
| 2026-02 | c_resumo_202602.pdf | True | False | 0 | 0 | True |
| 2026-03 | c_resumo_202603.pdf | True | False | 0 | 0 | True |
| 2026-04 | c_resumo_202604.pdf | True | False | 0 | 0 | True |
| 2026-05 | c_resumo_202605.pdf | True | False | 0 | 0 | True |
| 2026-06 | c_resumo_202606.pdf | True | False | 0 | 0 | True |
| 2026-07 | c_resumo_202607.pdf | True | True | 7 | 7 | True |
| 2026-08 | c_resumo_202608.pdf | True | True | 1 | 1 | True |
| 2026-09 | c_resumo_202609.pdf | True | True | 1 | 1 | True |
