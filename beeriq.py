"""
BeerIQ core library
-------------------
    from beeriq import BeerIQ
    biq = BeerIQ()
    biq.recommend(favorite="Sierra Nevada Pale Ale", n=10)
    biq.recommend(style="Stout", abv=(6, 9), sweet_bitter=+0.5, min_rating=3.8)
    biq.scouting_report("Sierra Nevada Pale Ale")
    biq.brewery_report("Sierra Nevada")
    biq.predict_rating(abv=6.5, style="American IPA", ibu=65)
    biq.analyze_flavor("Roasty, chocolatey and smooth with a hint of coffee bitterness.")
"""
import os, re, unicodedata, difflib, pickle
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data')
MODELS = os.path.join(ROOT, 'models')

FLAVORS = ['astringency', 'body', 'alcohol', 'bitter', 'sweet', 'sour', 'salty', 'fruits', 'hoppy', 'spices', 'malty']
FAMILY_WEIGHT = 1.6
DNA_LABELS = ['ABV', 'IBU'] + [f.capitalize() for f in FLAVORS]

FAMILIES = [  # (family, keywords) - first match wins
    ('Stout / Porter', ['stout', 'porter']),
    ('IPA', ['ipa', 'india pale']),
    ('Pale Ale', ['pale ale', 'bitter', 'esb']),
    ('Sour / Wild', ['lambic', 'gueuze', 'sour', 'wild', 'gose', 'berliner', 'flanders', 'faro']),
    ('Belgian', ['belgian', 'dubbel', 'tripel', 'quadrupel', 'saison', 'witbier', 'farmhouse', 'abbey', 'trappist']),
    ('Wheat', ['wheat', 'weizen', 'weisse', 'hefe', 'wit']),
    ('Strong / Barleywine', ['barleywine', 'barley wine', 'strong', 'old ale', 'wee heavy', 'scotch', 'imperial', 'doppelbock', 'eisbock']),
    ('Lager / Pilsner', ['lager', 'pilsner', 'pils', 'helles', 'bock', 'marzen', 'vienna', 'kolsch', 'dortmunder', 'schwarz', 'rauch']),
    ('Amber / Brown', ['amber', 'red ale', 'brown', 'mild', 'altbier', 'dunkel', 'irish']),
]


def norm(s):
    if pd.isna(s):
        return ''
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower().replace('&', ' and ')
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', ' ', s)).strip()


def style_family(style):
    s = str(style).lower()
    for fam, kws in FAMILIES:
        if any(k in s for k in kws):
            return fam
    return 'Other'


class BeerIQ:
    def __init__(self, data_dir=DATA, models_dir=MODELS):
        self.df = pd.read_csv(os.path.join(data_dir, 'beeriq_unified.csv'))
        self.brew = pd.read_csv(os.path.join(data_dir, 'beeriq_breweries.csv'))
        self.models_dir = models_dir
        df = self.df
        df['style_family'] = df['style'].map(style_family)
        df['key_full'] = (df.brewery_name.fillna('') + ' ' + df.beer_name).map(norm)
        self._build_dna()
        self._model = None
        self._lex = None

    # ------------------------------------------------------------------ Beer DNA
    def _build_dna(self):
        df = self.df
        self.dna_df = df[df.flavor_bitter.notna() & df.abv.notna()].copy()
        d = self.dna_df
        fl = d[['flavor_' + f for f in FLAVORS]].clip(lower=0).values.astype(float)
        share = np.sqrt(fl / np.maximum(fl.sum(1, keepdims=True), 1))          # flavor "shape", intensity-independent
        ibu = d.ibu.fillna((d.ibu_min + d.ibu_max) / 2).fillna(d.ibu.median()).values
        raw = np.column_stack([d.abv.values, ibu, share])
        self.dna_mu, self.dna_sd = raw.mean(0), raw.std(0) + 1e-9
        Z = (raw - self.dna_mu) / self.dna_sd
        self.dna_raw = raw
        fams = [f for f, _ in FAMILIES] + ['Other']
        self.fam_list = fams
        oh = (d['style_family'].to_numpy(dtype=object)[:, None] == np.array(fams, dtype=object)[None, :]).astype(float) * FAMILY_WEIGHT
        self.n_dna = Z.shape[1]
        Z = np.hstack([Z, oh])                      # style family is part of the DNA
        self.dna_z = Z
        self._Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        # 0-100 percentile per dimension (for bars/radars)
        self.dna_pct = pd.DataFrame(raw, columns=DNA_LABELS, index=d.index).rank(pct=True) * 100
        self.dna_df = d.reset_index(drop=True)
        self.dna_pct = self.dna_pct.reset_index(drop=True)

    # ------------------------------------------------------------------ lookup
    def find_beer(self, query, pool=None):
        """Best match for a free-text beer/brewery query, preferring beers with many reviews."""
        pool = self.df if pool is None else pool
        toks = norm(query).split()
        if not toks:
            return None
        m = pool[pool.key_full.apply(lambda k: all(t in k for t in toks))]
        if len(m):
            bn = m.beer_name.map(norm)
            cover = bn.map(lambda b: sum(t in b.split() or t in b for t in toks) / len(toks))
            exact = (m.key_full == norm(query)) | (bn == norm(query))
            m = m.assign(_s=exact.astype(float) * 10 + cover).sort_values(['_s', 'n_reviews'], ascending=False)
            return m.drop(columns='_s').iloc[0]
        keys = pool.key_full.tolist()
        c = difflib.get_close_matches(norm(query), keys, n=1, cutoff=0.6)
        return pool[pool.key_full == c[0]].iloc[0] if c else None

    # ------------------------------------------------------------------ recommender
    def recommend(self, favorite=None, style=None, abv=None, ibu=None, sweet_bitter=0.0,
                  min_rating=3.5, min_reviews=10, state=None, n=10, exclude_same_brewery=False):
        """
        favorite     : beer name (free text) to use as the DNA anchor
        style        : substring of style or style family to keep (e.g. 'Stout', 'IPA')
        abv, ibu     : (lo, hi) ranges; a scalar is treated as a target +/- tolerance
        sweet_bitter : -1 (sweet) ... +1 (bitter) tilts the query DNA
        min_rating   : minimum Bayesian rating
        Returns a dataframe of recommendations with a similarity score.
        """
        d, Zn = self.dna_df, self._Zn
        if not hasattr(d, 'style_family'):
            d['style_family'] = d['style'].map(style_family)
        self.last_anchor_note = None
        if favorite:
            row = self.find_beer(favorite)                      # search all 66k beers
            if row is None:
                raise ValueError(f'No beer matches "{favorite}"')
            hit = d.index[d.beer_id == row.beer_id]
            if len(hit):
                q = self.dna_z[hit[0]].copy()
            else:                                               # no flavor profile: use style centroid + own ABV/IBU
                peers = np.where(d['style'].to_numpy(dtype=object) == row['style'])[0]
                if len(peers) == 0:
                    peers = np.where(d.style_family.to_numpy(dtype=object) == style_family(row['style']))[0]
                q = self.dna_z[peers].mean(0).copy()
                if pd.notna(row.abv): q[0] = (row.abv - self.dna_mu[0]) / self.dna_sd[0]
                if pd.notna(row.ibu): q[1] = (row.ibu - self.dna_mu[1]) / self.dna_sd[1]
                self.last_anchor_note = (f'"{row.beer_name}" has no flavor profile; used the average {row["style"]} '
                                         f'flavor DNA with its own ABV/IBU.')
            anchor = row
        else:
            q = np.zeros(self.dna_z.shape[1]); anchor = None
        if abv is not None:
            t = abv if isinstance(abv, (tuple, list)) else (abv - .5, abv + .5)
            q[0] = ((np.mean(t)) - self.dna_mu[0]) / self.dna_sd[0]
        if ibu is not None:
            t = ibu if isinstance(ibu, (tuple, list)) else (ibu - 5, ibu + 5)
            q[1] = ((np.mean(t)) - self.dna_mu[1]) / self.dna_sd[1]
        if sweet_bitter:
            q[2 + FLAVORS.index('bitter')] += 1.5 * sweet_bitter
            q[2 + FLAVORS.index('sweet')] -= 1.5 * sweet_bitter
        if not np.linalg.norm(q):
            q[:] = 1e-6
        sims = Zn @ (q / (np.linalg.norm(q) + 1e-12))
        out = d.assign(similarity=np.clip((sims + 1) / 2, 0, 1) * 100)
        mask = (out.rating_bayes >= min_rating) & (out.n_reviews >= min_reviews)
        if anchor is not None:
            mask &= out.beer_id != anchor.beer_id
            if exclude_same_brewery:
                mask &= out.brewery_id != anchor.brewery_id
        if style:
            s = style.lower()
            mask &= out['style'].str.lower().str.contains(s) | out.style_family.str.lower().str.contains(s)
        if state:
            mask &= out.brewery_state == state
        for col, rng in (('abv', abv), ('ibu', ibu)):
            if rng is not None:
                lo, hi = rng if isinstance(rng, (tuple, list)) else (rng - (.5 if col == 'abv' else 5), rng + (.5 if col == 'abv' else 5))
                mask &= out[col].between(lo, hi)
        cols = ['beer_name', 'brewery_name', 'style', 'abv', 'ibu', 'rating_bayes', 'n_reviews', 'beer_score', 'similarity']
        res = out[mask].sort_values('similarity', ascending=False).head(n)[cols + ['beer_id']].reset_index(drop=True)
        res['similarity'] = res.similarity.round(1)
        return res

    def why_similar(self, a, b, top=3):
        """Dimensions where two beers are closest / furthest apart (in z-score units)."""
        ia = self.find_beer(a, self.dna_df).name; ib = self.find_beer(b, self.dna_df).name
        diff = pd.Series(np.abs(self.dna_z[ia][:self.n_dna] - self.dna_z[ib][:self.n_dna]), index=DNA_LABELS).sort_values()
        return {'closest': diff.head(top).index.tolist(), 'furthest': diff.tail(top).index.tolist()}

    # ------------------------------------------------------------------ scouting report
    def scouting_report(self, name, n_similar=3):
        row = self.find_beer(name)
        if row is None:
            raise ValueError(f'No beer matches "{name}"')
        peers = self.df[(self.df['style'] == row['style']) & (self.df.n_reviews >= 10)]
        pct = lambda col: float((peers[col] < row[col]).mean() * 100) if pd.notna(row[col]) and len(peers) else np.nan
        strengths, weaknesses = [], []
        for col, label in [('rating_aroma', 'Aroma'), ('rating_appearance', 'Appearance'), ('rating_palate', 'Mouthfeel'), ('rating_taste', 'Taste')]:
            p = pct(col)
            if p >= 75: strengths.append(f'{label}: top {max(1, 100 - p):.0f}% of style')
            elif p <= 25: weaknesses.append(f'{label}: bottom {max(1, p):.0f}% of style')
        if pd.notna(row.consistency):
            if row.consistency >= self.df.consistency.quantile(.75): strengths.append('Consistent ratings across reviewers')
            elif row.consistency <= self.df.consistency.quantile(.25): weaknesses.append('Divisive - ratings vary a lot')
        if row.n_reviews >= 200: strengths.append(f'Well-tested ({int(row.n_reviews)} reviews)')
        if row.n_reviews < 15: weaknesses.append(f'Thin evidence ({int(row.n_reviews)} reviews)')
        if pd.notna(row.abv) and row.abv >= 10: weaknesses.append(f'Very strong ({row.abv:.1f}% ABV)')
        rep = {'beer': row, 'strengths': strengths, 'weaknesses': weaknesses, 'style_percentile': pct('rating_bayes')}
        if row.beer_id in set(self.dna_df.beer_id):
            rep['similar'] = self.recommend(favorite=name, n=n_similar, min_rating=0, min_reviews=5)
        return rep

    def brewery_report(self, query, min_beers=1):
        m = self.brew[self.brew.brewery_name.fillna('').map(norm).str.contains(norm(query))]
        if m.empty:
            return None
        return m.sort_values('n_reviews', ascending=False).iloc[0]

    # ------------------------------------------------------------------ rating prediction
    def _load_model(self):
        if self._model is None:
            with open(os.path.join(self.models_dir, 'rating_model.pkl'), 'rb') as f:
                self._model = pickle.load(f)
        return self._model

    def predict_rating(self, abv, style, ibu=None, brewery=None, n_reviews=25, beer_age_years=1):
        """Predicted average BeerAdvocate 'overall' rating (1-5) for a hypothetical beer."""
        M = self._load_model()
        x = dict.fromkeys(M['columns'], 0.0)
        x['abv'] = abv
        x['log_reviews'] = np.log1p(n_reviews)
        x['beer_age_years'] = beer_age_years
        x['brewery_te'] = M['global_mean']; x['log_brewery_beers'] = 0.0
        if brewery:
            b = self.brewery_report(brewery)
            if b is not None:
                bb = self.df[(self.df.brewery_id == b.brewery_id) & (self.df.n_reviews >= 10)]
                k = 3
                x['brewery_te'] = (bb.rating_overall.sum() + k * M['global_mean']) / (len(bb) + k)
                x['log_brewery_beers'] = np.log1p((self.df.brewery_id == b.brewery_id).sum())
        col = 'style_' + style
        if col in x: x[col] = 1.0
        return float(M['model'].predict(pd.DataFrame([x])[M['columns']])[0])

    # ------------------------------------------------------------------ text flavor analyzer
    def _lexicon(self):
        if self._lex is None:
            import openpyxl
            path = os.path.join(ROOT, 'data', 'Beer_Descriptors_Simplified.xlsx')
            lex = {}
            wb = openpyxl.load_workbook(path, read_only=True)
            for ws in wb:
                rows = list(ws.iter_rows(values_only=True))
                heads = rows[0]
                for c in range(0, len(heads) - 1, 2):
                    cat = str(heads[c]).strip().lower()
                    for r in rows[1:]:
                        w, imp = r[c], r[c + 1] if c + 1 < len(r) else 1
                        if w and isinstance(w, str):
                            lex.setdefault(w.strip().lower(), []).append((cat, float(imp or 1)))
            self._lex = lex
        return self._lex

    def analyze_flavor(self, text):
        """Score free text (e.g. a review) on the 11 flavor descriptors using the descriptor lexicon."""
        lex = self._lexicon()
        scores = {}
        for w in re.findall(r"[a-z']+", text.lower()):
            for cat, imp in lex.get(w, []):
                scores[cat] = scores.get(cat, 0) + imp
        tot = sum(scores.values()) or 1
        return dict(sorted(((k, round(v / tot, 3)) for k, v in scores.items()), key=lambda kv: -kv[1]))
