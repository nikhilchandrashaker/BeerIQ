<p align="center"><img src="images/hero.png" alt="BeerIQ" width="100%"></p>

# BeerIQ — Beer Scouting & Recommendation Platform

Think *Baseball Savant + scouting report + recommendation engine*, but for beer.
BeerIQ merges **1,586,614 BeerAdvocate reviews** of **66,055 beers** with flavor profiles, IBU, brewery locations and US production data into one dataset, then builds on it:

| Module | What it does |
|---|---|
| **Beer DNA + recommender** | Content-based similarity on ABV, IBU, 11 flavor dimensions and style ("if you like X, try Y") |
| **Scouting reports** | Per-beer report cards: strengths, weaknesses, style percentile, similar beers |
| **Rating predictor** | Predict a beer's average rating from ABV, style, brewery record, etc. (4 models compared) |
| **What makes a great beer?** | ABV, style, flavor and sub-rating analysis, with correlation-vs-causation caveats |
| **Brewery scouting** | Quality, consistency, style specialization and innovation per brewery |
| **Geography & production** | US state maps; does production relate to popularity or quality? |
| **Flavor analyzer** | Scores free text against a beer-descriptor lexicon |

---

## Quick start

```bash
pip install pandas numpy scikit-learn scipy matplotlib openpyxl
cd src
python build_beeriq.py      # raw CSVs -> data/beeriq_unified.csv   (edit the upload path at the top)
python run_analysis.py      # all analyses, figures, model, results.json
python make_hero.py         # can + pint illustration
```

```python
from beeriq import BeerIQ
biq = BeerIQ()

biq.recommend(favorite="Guinness Draught", n=5)                    # similar beers
biq.recommend(style="Stout", abv=(6, 9), sweet_bitter=+0.5,        # preference-based
              min_rating=3.9)
biq.predict_rating(abv=6.5, style="American IPA", ibu=65, brewery="Stone")   # -> 4.03
biq.scouting_report("Zombie Dust")                                 # strengths / weaknesses / similar
biq.brewery_report("Hill Farmstead")                               # brewery scouting row
biq.analyze_flavor("Roasty, chocolatey and smooth with coffee bitterness")
```

---

## 1. Beer DNA & recommendations

Every beer with a flavor profile (3,057 beers) gets a **DNA vector**: ABV, IBU, the *shape* of its 11 flavor scores (astringency, body, alcohol, bitter, sweet, sour, salty, fruits, hoppy, spices, malty — normalised so review volume doesn't distort intensity), plus its style family. Similarity is cosine similarity on the standardised vector. Filters (style, ABV/IBU range, state, minimum rating) and a sweet↔bitter tilt let you steer results. Beers *without* a flavor profile are handled by anchoring on their style's average DNA plus their own ABV/IBU.

<p align="center"><img src="images/similar_beers.png" width="100%"></p>

Example: fans of **Russian River Brewing Company - Pliny The Elder** are pointed to Hop Ottin' IPA (98.6%), Hop Stoopid (98.3%), 2XIPA (98.1%), Arctic Panzer Wolf (98.0%). The two beers match most on *Salty, Hoppy, Bitter* and differ most on *Sweet, ABV, Malty*.

<p align="center"><img src="images/beer_dna_map.png" width="85%"></p>

The map above embeds all 3,057 profiled beers in 2-D with t-SNE. *Note:* style family is part of the DNA, so clusters partly reflect that by design; the within-cluster structure (e.g. the spread inside the Lager and "Other" groups) comes from the flavor profile.

## 2. Scouting report

<p align="center"><img src="images/beer_card.png" width="85%"></p>

Each beer also gets a **Beer Score (0–100)**: a percentile blend of Bayesian-adjusted rating (45%), rating relative to its own style (25%), reviewer agreement (15%) and review volume (15%). Top of the board (≥100 reviews):

| # | Beer | Brewery | Style | ABV | Rating | Reviews | Beer Score |
|---|---|---|---|---|---|---|---|
| 1 | Lunch | Maine Beer Company | American IPA | 7.0% | 4.33 | 183 | 97 |
| 2 | Abner Imperial IPA | Hill Farmstead Brewery | American Double / Imperial IPA | 8.2% | 4.43 | 110 | 97 |
| 3 | Heady Topper | The Alchemist | American Double / Imperial IPA | 8.0% | 4.63 | 469 | 97 |
| 4 | Zombie Dust | Three Floyds Brewing Co. & Brewpub | American Pale Ale (APA) | 6.2% | 4.51 | 393 | 96 |
| 5 | Citra DIPA | Kern River Brewing Company | American Double / Imperial IPA | 8.0% | 4.63 | 252 | 96 |
| 6 | Trappist Westvleteren 12 | Brouwerij Westvleteren (Sint-Sixtusabdij van Westvleteren) | Quadrupel (Quad) | 10.2% | 4.62 | 1272 | 96 |
| 7 | Founders CBS Imperial Stout | Founders Brewing Company | American Double / Imperial Stout | 10.6% | 4.59 | 637 | 96 |
| 8 | Parabola | Firestone Walker Brewing Co. | Russian Imperial Stout | 12.5% | 4.42 | 595 | 96 |
| 9 | Cantillon Blåbær Lambik | Brasserie Cantillon | Lambic - Fruit | 5.0% | 4.63 | 156 | 96 |
| 10 | Pure Hoppiness | Alpine Beer Company | American Double / Imperial IPA | 8.0% | 4.35 | 561 | 96 |

## 3. Predicting a beer's rating

Target: a beer's average overall rating (12,870 beers with ≥10 reviews and a valid ABV). Features: ABV, style (one-hot), brewery track record (smoothed average rating of the brewery's *other* beers, computed leave-one-out to avoid leakage), number of reviews, review span. 80/20 split.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear (Ridge) | 0.508 | 0.289 | 0.215 |
| Random Forest | 0.517 | 0.286 | 0.212 |
| Gradient Boosting | 0.522 | 0.285 | 0.211 |
| Neural Net (MLP) | 0.497 | 0.292 | 0.217 |
| *Baseline: style average* | 0.304 | 0.344 | 0.259 |

<p align="center"><img src="images/rating_model.png" width="100%"></p>

**What matters:** the brewery's track record dominates (shuffling it raises RMSE by 0.152, vs. 0.032 for style and 0.008 for ABV). Removing brewery information drops R² from 0.52 to 0.32; removing review count changes nothing (0.522), so the model isn't leaning on popularity. On the 2,671 beers with flavor profiles, adding the 11 flavor dimensions lifts 5-fold CV R² from 0.499 to 0.535 — a small but real gain.

Example predictions: `predict_rating(6.5, "American IPA", brewery="Stone")` → 4.03, versus 3.68 with no brewery information.

## 4. What makes a great beer?

**ABV.** Stronger beers rate higher overall (Spearman ρ = +0.29), but most of that is *style* — inside a style the relationship nearly vanishes (ρ = +0.07). Big styles tend to be rated highly; being big isn't what earns the rating.

<p align="center"><img src="images/rating_vs_abv.png" width="100%"></p>

**Style.** Highest-rated: American Wild Ale (4.04), Russian Imperial Stout (3.98), Gueuze (3.97), American Double / Imperial Stout (3.95), Flanders Red Ale (3.94). Lowest-rated: American Malt Liquor (2.53), Light Lager (2.81), Euro Strong Lager (2.86), American Adjunct Lager (2.97).

<p align="center"><img src="images/style_ratings.png" width="80%"></p>

**Flavor and sub-ratings.** Sub-ratings are extremely collinear with the overall score (Taste 0.95, Mouthfeel 0.93, Aroma 0.87, Appearance 0.82), so they can't be cleanly separated. Among raw correlations, ABV, IBU, spiciness and fruitiness go with higher ratings, while malty and astringent profiles go with lower ones. Once style is controlled everything shrinks to |ρ| ≤ 0.10: spiciness (+0.10) still helps, while sweetness (−0.10) and alcohol-forward flavor (−0.06) hurt.

<p align="center"><img src="images/flavor_drivers.png" width="100%"></p>

> **Correlation ≠ causation.** These are associations in observational review data. Reviewers self-select what they try, and enthusiasts rate rare, strong, hard-to-get beers more generously. Nothing here says a brewer should raise ABV to raise ratings.

**Description text.** Words in brewers' marketing descriptions barely predict ratings (cross-validated R² = 0.023); the "signal" words are mostly noise. The dataset has no review *text*, so review-level sentiment analysis isn't possible here.

<p align="center"><img src="images/description_words.png" width="70%"></p>

## 5. Brewery scouting

For breweries with ≥5 rated beers (n = 1,267; 245 qualify for the board with ≥8 beers and ≥1,000 reviews) BeerIQ computes: shrunken average rating (small samples pulled toward the mean), **consistency** (percentile of low rating spread), **style diversity**, **specialty style** (best style-adjusted performance), **innovation** (average rarity of the styles brewed) and a composite Brewery Score.

| # | Brewery | Rating | Consistency | Styles | Beers | Specialty |
|---|---|---|---|---|---|---|
| 1 | Cambridge Brewing Company | 3.94 | 81% | 34 | 57 | English Bitter |
| 2 | Minneapolis Town Hall Brewery | 3.98 | 68% | 55 | 157 | Russian Imperial Stout |
| 3 | Sly Fox Brewing Company | 3.92 | 83% | 37 | 58 | Maibock / Helles Bock |
| 4 | Brasserie d'Achouffe | 3.98 | 77% | 7 | 11 | Belgian IPA |
| 5 | East End Brewing Company | 3.99 | 64% | 25 | 38 | English Dark Mild Ale |
| 6 | AleSmith Brewing Company | 4.08 | 60% | 17 | 34 | American Porter |
| 7 | Stillwater Artisanal Ales | 3.90 | 91% | 7 | 24 | Belgian Strong Dark Ale |
| 8 | Hill Farmstead Brewery | 4.11 | 54% | 14 | 47 | American Porter |
| 9 | Brouwerij De Dolle Brouwers | 4.03 | 56% | 7 | 21 | Belgian Strong Dark Ale |
| 10 | Sierra Nevada Brewing Co. | 3.96 | 61% | 42 | 91 | American IPA |

<p align="center"><img src="images/brewery_scouting.png" width="100%"></p>

## 6. Geography & production

<p align="center"><img src="images/state_map.png" width="100%"></p>

Rating data has no country field, so the geographic view covers the **US only** (4,110 beers linked to a state via the Craft Cans data). Production is from state barrel totals.

<p align="center"><img src="images/production_vs_popularity.png" width="100%"></p>

Across 34 states, production has only a weak relationship with attention (ρ = +0.12) and quality (ρ = +0.11). At the beer level, more-reviewed beers rate higher (avg 3.69 for 10–19 reviews vs. 3.91 for 500+) — but that partly reflects survivorship: good beers attract reviews, so review count is a *consequence* of quality and shouldn't be read as a cause.

---

## Data & method

**Sources** — BeerAdvocate reviews (`beer_reviews.csv`; `beer_reviews-2.csv` is an identical copy plus an index column), beer profiles & flavor scores (`beer_profile_and_ratings.csv`, `beer_data_set.csv`), Craft Cans (`beers.csv`, `breweries.csv`), state production (`beer_states.csv`), brewer-size context, name fuzzy-match lists, and a descriptor lexicon (`Beer_Descriptors_Simplified.xlsx`).

**Merge** — One row per BeerAdvocate beer. Profiles joined through the supplied fuzzy-match list (3,197 / 3,197). `beer_data_set` and Craft Cans joined on normalised brewery+beer name with a TF-IDF character n-gram fallback at cosine ≥ 0.90 (I raised this from 0.80 after spot-checks showed false matches like *Full Nelson* → *Half Nelson*). Unmatched rows are left blank, not guessed.

**Coverage** (of 66,055 beers) — flavor profile 3,197 (3,057 also have a valid ABV and form the DNA set), ≥10 reviews 14,228, US state 8,372, measured IBU 380 (IBU otherwise falls back to the style-range midpoint, flagged in `ibu_source`).

### Key columns in `data/beeriq_unified.csv`

| Column | Meaning |
|---|---|
| `beer_id`, `beer_name`, `brewery_id`, `brewery_name`, `style`, `style_family` | Identity |
| `abv`, `abv_raw`, `ibu`, `ibu_source` | ABV (implausible values <0.5% or >25% blanked; raw kept), best-available IBU |
| `n_reviews`, `n_reviewers`, `rating_overall`, `rating_aroma/appearance/palate/taste` | Review aggregates |
| `rating_bayes` | Rating shrunk toward the global mean (prior weight = 10 reviews) |
| `style_adj_rating` | z-score of `rating_bayes` within style |
| `rating_std`, `consistency` | Reviewer spread; `1 − std/5` |
| `beer_score` | Composite 0–100 score |
| `flavor_*`, `description`, `ibu_min/max` | Flavor-profile dataset |
| `brewery_state`, `brewery_city`, `country`, `state_barrels_latest`, `state_barrels_growth_since_2008` | US location and state production |

### Project layout

```
BeerIQ/
├── README.md
├── results.json            # every number quoted above
├── data/                   # unified beer table, brewery table, state table, context tables
├── images/                 # figures used in this README
├── models/rating_model.pkl # trained gradient-boosting predictor
└── src/                    # build_beeriq.py, beeriq.py (library), run_analysis.py, make_hero.py, make_readme.py
```

## Limitations

- **Old data.** Reviews span 1996–2012; ratings reflect that era, and many breweries have changed or closed. "Beer age" is the span between first and last review, not release date.
- **Coverage is uneven.** Only ~5% of beers have flavor profiles and ~13% a US state; flavor-based features and the maps describe those subsets, which skew American and popular.
- **Single-source ratings.** BeerAdvocate reviewers are enthusiasts, not the general public.
- **National production tables** (`beer_taxed`, `brewing_materials`) are monthly aggregates and can't be joined to individual beers, so they aren't used per-beer.
- **Brewery effect.** The rating model's strength comes largely from brewery reputation; for a brand-new brewery it falls back to style and ABV (R² ≈ 0.32).
