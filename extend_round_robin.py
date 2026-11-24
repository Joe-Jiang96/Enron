"""Extend fixed broad-relevance round-robin evaluation to all 1920 emails."""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis_outputs/round_robin_extended'
ORIGINAL_SEED = 6454
SEED = 6455
SEMANTIC_REASSESSMENT_FILE = 'semantic_reassessment_changes.csv'
SEMANTIC_TOP200_REVIEW_FILE = 'semantic_top200_review.csv'
LABEL_FIELDS = ['relevant', 'tier', 'uncertain', 'rationale']
RUBRIC = ('Broad investigative-topic relevance, applied identically across ranks and sources: '
          '1 for substantive or indirect connections to SPEs, off-balance-sheet or structured '
          'financing, associated transactions/documents/participants, retained obligations or '
          'financial accounting/disclosure questions bearing on the investigation. Explicitly '
          'identified target-transaction document requests and administrative leads qualify. '
          'Neither a contradiction nor knowledge of wrongdoing is required. Generic trading '
          'contracts, routine customer credit/guarantees, unrelated market misconduct, social '
          'messages and generic Enron news without such a connection receive 0. '
          'Insufficient context is coded 0 with an uncertainty flag, not assumed relevant.')
REASSESSMENTS = (
    ('middle_reassessment', 1001, 1300),
    ('tail_reassessment', 1300, 1920),
)
SEMANTIC_BAND_ENDS = (1200, 1500, 1600)
SEMANTIC_BAND_REVIEW_FILE = 'semantic_band_reassessment_changes.csv'
RANK_1500_REVIEW_FILE = 'band_1401_1500_reassessment_changes.csv'


def save(d,name):
    d.to_parquet(OUT/f'{name}.parquet',index=False)
    d.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')


def draw_review_sample(combined):
    original = combined.combined_rank.le(1000)
    cohorts = ((combined.loc[original], ORIGINAL_SEED, 'R'),
               (combined.loc[~original], SEED, 'E'))
    draws = []
    for pool, seed, prefix in cohorts:
        rng = np.random.default_rng(seed)
        selected = []
        for _, group in pool.groupby(['band_end', 'first_source'], sort=True):
            count = min(20, len(group))
            positions = np.sort(rng.choice(len(group), count, replace=False))
            chosen = group.iloc[positions].copy()
            chosen['stratum_population'] = len(group)
            chosen['sampling_probability'] = count / len(group)
            chosen['sampling_weight'] = len(group) / count
            selected.append(chosen)
        sample = pd.concat(selected).reset_index(drop=True)
        sample.insert(0, 'review_id', [f'{prefix}{i:03d}' for i in range(1, len(sample) + 1)])
        draws.append(sample)
    return pd.concat(draws, ignore_index=True)


def prepare():
    plan = json.loads((OUT / 'review_plan.json').read_text(encoding='utf-8'))
    ranking_path = OUT / 'combined_deduplicated.parquet'
    assert hashlib.sha256(ranking_path.read_bytes()).hexdigest() == plan['combined_sha256']
    combined = pd.read_parquet(ranking_path)
    selected = draw_review_sample(combined)
    keys = ['review_id', 'email_id', 'combined_rank', 'first_source']
    sample_signature = selected[keys].to_csv(index=False).encode()
    assert hashlib.sha256(sample_signature).hexdigest() == plan['sample_sha256']

    sample_path = OUT / 'review_sample.parquet'
    if not sample_path.exists():
        sample_path = OUT / 'sample_relevancy_verdicts.parquet'
    saved = pd.read_parquet(sample_path).drop(columns=LABEL_FIELDS, errors='ignore')
    pd.testing.assert_frame_equal(selected, saved[selected.columns])
    save(saved, 'review_sample')
    print(json.dumps(plan, indent=2))
    print(saved.groupby(['band_end', 'first_source']).agg(
        population=('stratum_population', 'first'),
        sampled=('email_id', 'size'),
    ).to_string())


def total_bounds(r, n, N, alpha=.05):
    """Equal-tailed exact finite-population interval by hypergeometric inversion."""
    totals = np.arange(r, N-n+r+1)
    accepted = (hypergeom.cdf(r, N, totals, n) >= alpha/2) & (hypergeom.sf(r-1, N, totals, n) >= alpha/2)
    return int(totals[accepted].min()), int(totals[accepted].max())


def elbow(ks, totals):
    # Maximum vertical distance above endpoint chord on normalized cumulative curve.
    x = np.r_[0, ks]/ks[-1]
    y = np.r_[0, totals]
    if y[-1] <= 0:
        return None, 0.
    gap = y/y[-1]-x
    j = int(np.argmax(gap))
    return (int(np.r_[0, ks][j]) if j not in (0,len(x)-1) and gap[j]>1e-10 else None), float(gap[j])


def read(start,end):
    sample=pd.read_parquet(OUT/'review_sample.parquet')
    done=pd.read_csv(OUT/'assessments.tsv',sep='\t').review_id
    for r in sample.loc[~sample.review_id.isin(done)].iloc[start-1:end].itertuples():
        text=re.sub(r'(?m)^\s*(?:From:|To:|Cc:|cc:|Sent:|Subject:|-----|>).*$', '',r.body_clean)
        text=re.sub(r'\n{3,}','\n\n',text)
        if r.prior_narrow_rationale:
            print(f'\n=== {r.review_id} | {r.subject} | PRIOR NARROW {r.prior_narrow_verdict} ===\n{r.prior_narrow_rationale}\nOpening context: {text[:500]}')
        else:
            if len(text)>2500:
                chunks=re.split(r'(?<=[.!?])\s+|\n\s*\n',text)
                matches=[i for i,p in enumerate(chunks) if re.search(r'raptor|ljm|jedi|chewco|whitewing|condor|osprey|mahonia|yosemite|special[ -]purpose|\bSPE\b|off[ -]balance|prepay|retained|accounting|disclos',p,re.I)]
                text='[SCREENING EXCERPTS]\n'+text[:700]+'\n[...]\n'+'\n[...]\n'.join(' '.join(chunks[max(0,i-1):i+2])[:950] for i in matches[:3])
                text=text[:2700]
            print(f'\n=== {r.review_id} | {r.subject} | full chars {len(r.body_clean)} ===\n{text}')


def statistics(reviewed):
    rows=[]
    populations=reviewed.groupby(['band_end','first_source']).stratum_population.first().groupby('band_end').sum()
    n_strata=reviewed.groupby(['band_end','first_source']).ngroups
    for (end,source),g in reviewed.groupby(['band_end','first_source']):
        N=int(g.stratum_population.iloc[0]);r=int(g.relevant.sum());n=len(g)
        lo,hi=total_bounds(r,n,N)
        blo,bhi=total_bounds(r,n,N,alpha=.05/2)
        clo,chi=total_bounds(r,n,N,alpha=.05/n_strata)
        actual_end=int(end-100+populations[end])
        rows.append(dict(k=actual_end,band=f'{end-99}-{actual_end}',source=source,population=N,sampled=n,relevant=r,
            uncertain=int(g.uncertain.sum()),precision=r/n,estimated_relevant=N*r/n,
            precision_ci_low=lo/N,precision_ci_high=hi/N,band_joint_low=blo,band_joint_high=bhi,
            cumulative_joint_low=clo,cumulative_joint_high=chi))
    strata=pd.DataFrame(rows).sort_values(['source','k'])
    for col in ['estimated_relevant','population']:
        strata[f'cumulative_{col}']=strata.groupby('source')[col].cumsum()
    strata['cumulative_precision']=strata.cumulative_estimated_relevant/strata.cumulative_population
    combo=strata.groupby(['k','band'],as_index=False).agg(population=('population','sum'),sampled=('sampled','sum'),
        relevant=('relevant','sum'),uncertain=('uncertain','sum'),estimated_relevant=('estimated_relevant','sum'),
        band_joint_low=('band_joint_low','sum'),band_joint_high=('band_joint_high','sum'),
        cumulative_joint_low=('cumulative_joint_low','sum'),cumulative_joint_high=('cumulative_joint_high','sum'))
    combo['source']='combined';combo['precision']=combo.estimated_relevant/combo.population
    combo['unweighted_sample_precision']=combo.relevant/combo.sampled
    combo['precision_ci_low']=combo.band_joint_low/combo.population;combo['precision_ci_high']=combo.band_joint_high/combo.population
    combo['cumulative_estimated_relevant']=combo.estimated_relevant.cumsum()
    combo['cumulative_population']=combo.population.cumsum()
    combo['cumulative_precision']=combo.cumulative_estimated_relevant/combo.k
    combo['cumulative_simultaneous_low']=combo.cumulative_joint_low.cumsum()
    combo['cumulative_simultaneous_high']=combo.cumulative_joint_high.cumsum()
    result=pd.concat([combo,strata],ignore_index=True)
    summary={}
    for source in ['combined','semantic','tfidf']:
        b=result[result.source.eq(source)].sort_values('k')
        candidate,gap=elbow(b.k.to_numpy(),b.cumulative_estimated_relevant.to_numpy())
        original=b[b.k.le(1000)].iloc[-1]
        summary[source]=dict(first_band_precision=float(b.precision.iloc[0]),sampled=int(b.sampled.sum()),
            relevant=int(b.relevant.sum()),estimated_relevant_through_1000=float(original.cumulative_estimated_relevant),
            estimated_precision_through_1000=float(original.cumulative_precision),
            max_k=int(b.k.iloc[-1]),estimated_relevant_through_endpoint=float(b.estimated_relevant.sum()),
            estimated_precision_through_endpoint=float(b.cumulative_precision.iloc[-1]),
            geometric_elbow_k=candidate,normalized_gap=gap,
            elbow_coordinate='Combined rank K; maximum normalized cumulative distance above endpoint chord')
        if candidate:
            chosen=b[b.k.eq(candidate)].iloc[0]
            summary[source]['estimated_relevant_at_elbow']=float(chosen.cumulative_estimated_relevant)
            summary[source]['estimated_precision_at_elbow']=float(chosen.cumulative_precision)
            summary[source]['estimated_relevant_after_elbow']=float(b.cumulative_estimated_relevant.iloc[-1]-chosen.cumulative_estimated_relevant)
    return result,summary


def apply_label_audit(reviewed, audit):
    reviewed = reviewed.set_index('review_id')
    assert reviewed.loc[audit.index, 'email_id'].eq(audit.email_id).all()
    before = audit[[f'before_{field}' for field in LABEL_FIELDS]].set_axis(LABEL_FIELDS, axis=1)
    after = audit[[f'after_{field}' for field in LABEL_FIELDS]].set_axis(LABEL_FIELDS, axis=1)
    pd.testing.assert_frame_equal(reviewed.loc[audit.index, LABEL_FIELDS], before)
    reviewed.loc[audit.index, LABEL_FIELDS] = after
    return reviewed.reset_index()


def apply_semantic_reassessment(reviewed):
    audit_path = OUT / SEMANTIC_REASSESSMENT_FILE
    if not audit_path.exists():
        return reviewed, {}

    audit = pd.read_csv(audit_path, keep_default_na=False).set_index('review_id', verify_integrity=True)
    scope = reviewed.first_source.eq('semantic') & reviewed.semantic_rank.between(1, 500)
    scoped = reviewed.loc[scope]
    assert audit.index.isin(scoped.review_id).all()
    assert audit.before_relevant.eq(0).all() and audit.after_relevant.eq(1).all()
    summary = {
        'sampled': len(scoped),
        'reviewed_prior_negatives': int(scoped.relevant.eq(0).sum()),
        'relevant_before': int(scoped.relevant.sum()),
        'relevant_after': int(scoped.relevant.sum()) + len(audit),
        'reclassified_as_relevant': len(audit),
    }
    return apply_label_audit(reviewed, audit), summary


def semantic_reassessment_note(summary, audit_link):
    review = summary.get('semantic_reassessment')
    if not review:
        return ''
    return (
        f'<p>The targeted second review examined all {review["reviewed_prior_negatives"]} previously negative judgments '
        f'among {review["sampled"]} sampled emails attributed to semantic search with original semantic ranks 1–500. '
        'The review considered concrete financing arrangements, where obligations or risks remained, '
        'and how positions or financial results were reported. '
        f'The relevant count changed from {review["relevant_before"]} to {review["relevant_after"]}. '
        f'<a href="{audit_link}">See before-and-after verdicts and reasons.</a></p>'
    )


def apply_semantic_top200_review(reviewed):
    review_path = OUT / SEMANTIC_TOP200_REVIEW_FILE
    if not review_path.exists():
        return reviewed, {}

    top200 = pd.read_csv(review_path, keep_default_na=False)
    assert top200.semantic_rank.tolist() == list(range(1, 201))
    assert top200.email_id.is_unique and top200.relevant.isin([0, 1]).all()
    overlap = top200.loc[top200.review_id.ne('')].set_index('review_id', verify_integrity=True)
    assert set(overlap.index) == set(reviewed.loc[reviewed.semantic_rank.between(1, 200), 'review_id'])
    audit = overlap.rename(columns={field: f'after_{field}' for field in LABEL_FIELDS}).astype(
        {'before_relevant': 'int64', 'before_uncertain': 'int64'}
    )
    return apply_label_audit(reviewed, audit), {
        'reviewed': len(top200),
        'relevant': int(top200.relevant.sum()),
        'sample_overlap': len(audit),
        'sample_reclassified_as_relevant': int(audit.after_relevant.gt(audit.before_relevant).sum()),
        'sample_reclassified_as_unrelated': int(audit.after_relevant.lt(audit.before_relevant).sum()),
    }


def semantic_top200_note(summary, review_link):
    review = summary.get('semantic_top200_review')
    if not review:
        return ''
    return (
        f'<p>A later full review found {review["relevant"]} broadly relevant emails at original semantic ranks '
        f'1–{review["reviewed"]}. It updated the verdicts of overlapping samples and kept the original draws '
        f'for the round-robin estimates. <a href="{review_link}">See every top-200 verdict and reason.</a></p>'
    )


def apply_reassessment(reviewed, audit_file, scope):
    audit_path = OUT / audit_file
    if not audit_path.exists():
        return reviewed, {}

    audit = pd.read_csv(audit_path, keep_default_na=False).set_index('review_id', verify_integrity=True)
    assert audit.index.isin(reviewed.loc[scope, 'review_id']).all()
    assert audit.after_relevant.isin([0, 1]).all()
    assert audit.before_relevant.ne(audit.after_relevant).all()
    updated = apply_label_audit(reviewed, audit)
    return updated, {
        'sampled': int(scope.sum()),
        'relevant_before': int(reviewed.loc[scope, 'relevant'].sum()),
        'relevant_after': int(updated.loc[scope, 'relevant'].sum()),
        'reclassified_as_unrelated': int(audit.after_relevant.eq(0).sum()),
        'reclassified_as_relevant': int(audit.after_relevant.eq(1).sum()),
    }


def evaluate():
    sample=pd.read_parquet(OUT/'review_sample.parquet')
    labels=pd.read_csv(OUT/'assessments.tsv',sep='\t',keep_default_na=False)
    assert labels.review_id.is_unique and set(labels.review_id)==set(sample.review_id)
    assert labels.relevant.isin([0,1]).all() and labels.uncertain.isin([0,1]).all()
    assert labels.rationale.str.len().gt(0).all()
    plan = json.loads((OUT / 'review_plan.json').read_text(encoding='utf-8'))
    ledger_lines = (OUT / 'assessments.tsv').read_bytes().splitlines(keepends=True)
    original_labels = ledger_lines[0] + b''.join(line for line in ledger_lines[1:] if line.startswith(b'R'))
    assert hashlib.sha256(original_labels).hexdigest() == plan['original_labels_sha256']
    reviewed=sample.merge(labels,on='review_id',validate='one_to_one')
    reviewed, semantic_review = apply_semantic_reassessment(reviewed)
    reviewed, top200_review = apply_semantic_top200_review(reviewed)
    reviews = {}
    if semantic_review:
        reviews['semantic_reassessment'] = semantic_review
    if top200_review:
        reviews['semantic_top200_review'] = top200_review
    ranking = pd.read_parquet(OUT / 'combined_deduplicated.parquet', columns=['combined_rank'])
    for review_key, first_rank, last_rank in REASSESSMENTS:
        audit_path = OUT / f'{review_key}_changes.csv'
        if not audit_path.exists():
            continue
        audit = pd.read_csv(audit_path)
        scoped = reviewed.loc[reviewed.combined_rank.between(first_rank, last_rank)]
        changed = scoped.merge(audit[['review_id', 'before_relevant', 'after_relevant']],
                               on='review_id', validate='one_to_one')
        assert len(changed) == len(audit)
        assert changed.relevant.eq(changed.after_relevant).all()
        demoted = changed.before_relevant.gt(changed.after_relevant)
        promoted = changed.before_relevant.lt(changed.after_relevant)
        population = int(ranking.combined_rank.between(first_rank, last_rank).sum())
        estimate = float((scoped.relevant * scoped.sampling_weight).sum())
        previous_estimate = estimate + float(
            ((changed.before_relevant - changed.after_relevant) * changed.sampling_weight).sum()
        )
        reviews[review_key] = {
            'first_rank': first_rank,
            'last_rank': last_rank,
            'sampled': len(scoped),
            'relevant_before': int(scoped.relevant.sum() + demoted.sum() - promoted.sum()),
            'relevant_after': int(scoped.relevant.sum()),
            'reclassified_as_unrelated': int(demoted.sum()),
            'reclassified_as_relevant': int(promoted.sum()),
            'estimated_relevant_before': previous_estimate,
            'estimated_relevant_after': estimate,
            'weighted_precision_before': previous_estimate / population,
            'weighted_precision_after': estimate / population,
        }
    # Apply the later corrections in review order so earlier results remain auditable.
    semantic_scope = reviewed.first_source.eq('semantic') & reviewed.band_end.isin(SEMANTIC_BAND_ENDS)
    reviewed, semantic_band_review = apply_reassessment(reviewed, SEMANTIC_BAND_REVIEW_FILE, semantic_scope)
    if semantic_band_review:
        reviews['semantic_band_reassessment'] = semantic_band_review
    rank_1500_scope = reviewed.band_end.eq(1500) & reviewed.relevant.eq(1)
    reviewed, rank_1500_review = apply_reassessment(reviewed, RANK_1500_REVIEW_FILE, rank_1500_scope)
    if rank_1500_review:
        reviews['band_1401_1500_reassessment'] = rank_1500_review
    result,summary=statistics(reviewed)
    summary.update(reviews)
    save(reviewed,'sample_relevancy_verdicts')
    cols=['review_id','email_id','combined_rank','band_end','first_source','source_rank','semantic_rank','lexical_rank',
        'stratum_population','sampling_probability','sampling_weight','subject','date_utc','relevant','tier','uncertain','rationale']
    reviewed[cols].to_csv(OUT/'sample_verdicts_compact.csv',index=False,encoding='utf-8-sig')
    save(result,'precision_by_band_and_source')
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot(result,summary);report(reviewed,result,summary)
    plot_combined_charts(dual_source_estimates())
    print(result[result.source.eq('combined')][['band','population','sampled','relevant','precision','cumulative_estimated_relevant']].to_string(index=False));print(json.dumps(summary,indent=2))


def plot(df,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    chart_max_rank = 1800
    chart_data = df.loc[df.k.le(chart_max_rank)]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(18,9),sharex=True)
    for i,(source,name,color) in enumerate([('combined','Combined (population weighted)','#315f9a'),('semantic','Semantic-attributed emails','#b4643e'),('tfidf','TF-IDF-attributed emails','#12827c')]):
        b=chart_data[chart_data.source.eq(source)].sort_values('k');k=b.k.to_numpy();p=b.precision.to_numpy()
        axs[0,i].errorbar(k,p,yerr=np.array([p-b.precision_ci_low,b.precision_ci_high-p]),fmt='o-',color=color,capsize=3,lw=1.6,markersize=4)
        axs[0,i].set(title=name,ylabel='Broad-topic band precision',ylim=(0,1.03))
        axs[0,i].yaxis.set_major_formatter(PercentFormatter(1))
        axs[1,i].plot(np.r_[0,k],np.r_[0,b.cumulative_estimated_relevant],'-o',color=color,lw=1.8,markersize=4)
        if source=='combined':axs[1,i].fill_between(np.r_[0,k],np.r_[0,b.cumulative_simultaneous_low],np.r_[0,b.cumulative_simultaneous_high],color=color,alpha=.12,label='Conservative simultaneous 95% bounds')
        candidate=summary[source]['geometric_elbow_k']
        if candidate and candidate <= chart_max_rank:
            axs[1,i].axvline(candidate,ls=':',color='#6b7280',label=f'Full-pool elbow K={candidate}')
        axs[1,i].set(title='Estimated cumulative relevant emails',xlabel='Deduplicated combined rank K',ylabel='Estimated count',ylim=(0,None),xlim=(0,chart_max_rank))
        if axs[1,i].get_legend_handles_labels()[0]:
            axs[1,i].legend(loc='upper left',fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.17);ax.set_xticks(np.arange(200,chart_max_rank+1,200));ax.tick_params(axis='x',rotation=45)
        ax.axvline(1000,color='#94a3b8',ls='--',alpha=.7,lw=1)
    fig.suptitle(f'Round-robin relevance through combined rank {chart_max_rank:,}',fontsize=19,x=.06,ha='left')
    visible_reviews = chart_data.loc[chart_data.source.eq('combined'), 'sampled'].sum()
    fig.text(.06,.925,f'{visible_reviews} reviewed emails across the displayed bands | 20 random emails per source and band',color='#475569')
    reviewed_ranges = [
        f'{start}–{min(end, chart_max_rank)}'
        for key, start, end in REASSESSMENTS if key in summary and start <= chart_max_rank
    ]
    if all(key in summary for key, _, _ in REASSESSMENTS):
        reviewed_ranges = [f'{REASSESSMENTS[0][1]}–{chart_max_rank}']
    review_note = f' Samples at ranks {", ".join(reviewed_ranges)} received a second review.' if reviewed_ranges else ''
    semantic_note = ' Second review included original semantic ranks 1–500.' if 'semantic_reassessment' in summary else ''
    fig.text(.06,.017,'Source panels use combined ranks.' + semantic_note + review_note +
             '\nThe dashed line marks K=1000. Elbows and simultaneous bounds use the full 1,920-email analysis.',
             fontsize=10,color='#475569')
    fig.tight_layout(rect=(0,.07,1,.91));fig.savefig(OUT/'precision_combined_and_sources_extended.png',dpi=180);plt.close(fig)


def dual_source_estimates():
    ranking = pd.read_parquet(OUT / 'combined_deduplicated.parquet').sort_values(
        ['in_both_sources', 'combined_rank'], ascending=[False, True]
    )
    sample = pd.read_parquet(OUT / 'sample_relevancy_verdicts.parquet',
                             columns=['email_id', 'relevant', 'sampling_weight', 'band_end',
                                      'first_source', 'stratum_population']).rename(
        columns={'band_end': 'original_band_end'}
    )
    ranked = ranking[['email_id', 'in_both_sources']].merge(
        sample, on='email_id', how='left', validate='one_to_one', sort=False
    )
    ranks = np.arange(1, len(ranked) + 1)
    ranked['band_end'] = ((ranks - 1) // 100 + 1) * 100
    ranked['weighted_relevant'] = ranked.relevant * ranked.sampling_weight
    groups = ranked.groupby(['band_end', 'in_both_sources']).agg(
        pop=('email_id', 'size'),
        weight=('sampling_weight', 'sum'),
        weighted_relevant=('weighted_relevant', 'sum'),
    )
    # Calibrate the originally weighted sample to each new band's actual source overlap.
    assert groups.weight.gt(0).all(), 'A reranked group has no sampled verdicts'
    groups['relevance_rate'] = groups.weighted_relevant / groups.weight
    groups['estimated_relevant'] = groups['pop'] * groups.relevance_rate
    bands = groups.groupby(level='band_end')[['pop', 'estimated_relevant']].sum()
    bands['k'] = bands['pop'].cumsum()
    bands['precision'] = bands.estimated_relevant / bands['pop']
    bands['cumulative_estimated_relevant'] = bands.estimated_relevant.cumsum()

    observed = ranked.loc[ranked.relevant.notna()].join(
        groups[['pop', 'weight', 'relevance_rate']], on=['band_end', 'in_both_sources']
    )
    strata = observed.groupby(['original_band_end', 'first_source'])
    n = strata.size()
    population = strata.stratum_population.first()
    variance_factor = population**2 * (1 - n / population) / n
    # Linearize the calibrated ratios, then measure sampling variance in the
    # original strata. Each band's influence is zero for draws outside that band.
    influence = observed['pop'] * (observed.relevant - observed.relevance_rate) / observed.weight
    standard_errors = []
    for band_end, population_in_band in bands['pop'].items():
        band_influence = influence.where(observed.band_end.eq(band_end), 0)
        variance = band_influence.groupby(
            [observed.original_band_end, observed.first_source]
        ).var()
        standard_errors.append(np.sqrt((variance_factor * variance).sum()) / population_in_band)
    margin = 1.96 * np.asarray(standard_errors)
    bands['precision_ci_low'] = (bands.precision - margin).clip(lower=0)
    bands['precision_ci_high'] = (bands.precision + margin).clip(upper=1)
    return bands.reset_index(drop=True)


def plot_combined_charts(data, review_cutoff=1300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    bands = data.loc[data.k.le(1800)].sort_values('k')
    ranks = bands.k.to_numpy()
    ticks = np.arange(0, ranks[-1] + 1, 200)
    blue = '#315f9a'

    with plt.rc_context({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.axisbelow': True}):
        fig, ax = plt.subplots(figsize=(10, 5.6), layout='constrained')
        ax.set_title('What share of emails in each rank band is relevant?', loc='left',
                     fontsize=17, fontweight='bold', pad=28)
        ax.text(0, 1.012, 'Emails found by both searches are ranked first',
                transform=ax.transAxes, color='#475569', fontsize=10)
        precision = bands.precision.to_numpy()
        errors = [precision - bands.precision_ci_low, bands.precision_ci_high - precision]
        ax.errorbar(ranks, precision, yerr=errors, fmt='o-', color=blue, lw=1.8,
                    markersize=4.5, capsize=4, label='Estimate and approximate 95% confidence interval')
        ax.set(xlabel='Combined rank at the end of each band',
               ylabel='Estimated share of relevant emails',
               xlim=(0, ranks[-1] + 30), ylim=(0, 1.03), xticks=ticks[1:])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper right', frameon=False, fontsize=9)
        fig.savefig(OUT / 'combined_precision_by_rank.png', dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6.2), layout='constrained')
        ax.set_title('How many relevant emails are in the top K?', loc='left',
                     fontsize=17, fontweight='bold', pad=28)
        ax.text(0, 1.012, 'Dual-source matches first · cumulative estimate as the review budget grows',
                transform=ax.transAxes, color='#475569', fontsize=10)
        k = np.r_[0, ranks]
        relevant = np.r_[0, bands.cumulative_estimated_relevant]
        ax.fill_between(k, relevant, k, color='#f1cf83', alpha=.38,
                        label='Gap: estimated non-relevant emails')
        ax.plot(k, k, '-', color='black', lw=1.5, label='All reviewed emails relevant (y = x)')
        ax.plot(k, relevant, 'o-', color=blue, lw=2.2, markersize=4.5,
                label='Estimated relevant emails')
        if review_cutoff <= ranks[-1]:
            ax.axvline(review_cutoff, color='#773e86', ls='-.', lw=1.6,
                       label=f'Selected review cutoff (K = {review_cutoff:,})')
        ax.set(xlabel='Number of top-ranked emails reviewed (K)', ylabel='Cumulative email count',
               xlim=(0, ranks[-1] + 30), ylim=(0, ranks[-1] + 80), xticks=ticks, yticks=ticks)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        fig.savefig(OUT / 'combined_cumulative_relevant_emails.png', dpi=180)
        plt.close(fig)


def report(reviewed,df,summary):
    c = summary['combined']
    first_horizon = df.loc[df.source.eq('combined') & df.k.le(1000)]
    original_elbow, _ = elbow(first_horizon.k.to_numpy(), first_horizon.cumulative_estimated_relevant.to_numpy())
    doc='''<!doctype html><html><head><meta charset="utf-8"><title>Extended round-robin precision</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:35px auto;padding:0 24px;color:#243746}img{width:100%}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;font:14px/1.6 system-ui}details{padding:10px;border-bottom:1px solid #ddd}summary{cursor:pointer}</style></head><body><h1>Round-robin relevance through K=1920</h1>'''
    doc+=f'<p><b>Rubric:</b> {html.escape(RUBRIC)}</p>'
    doc+='<p>Semantic max(who, how much, when) top 1000 and TF-IDF top 1000 alternate S1,L1,...; first-email-ID deduplication reduces 2000 slots to 1920 emails. Original source scores and the documented TF-IDF supplied-rank correction are retained. There is no definition-focused reranking. Source attribution follows first occurrence, so source panels are contributions within combined bands rather than standalone source-rank evaluations.</p>'
    doc+='<p>The original 400 sampled emails through combined K=1000 are retained. Nine further full bands each receive 20 random emails per source, adding 360 reviews with extension seed 6455. Nominal band 1901-2000 contains only ranks 1901-1920; both source groups are fully reviewed, adding 20. Total: 780 unique reviewed emails. Ranks 1921-2000 do not exist and are not assigned zero precision. Different email IDs with identical text remain separate email observations.</p>'
    doc+='<p>For every source stratum, p=relevant/n and estimated relevant=N*p. Combined precision sums estimates and divides by the actual band population (100 or 20). The final census has no sampling uncertainty conditional on labels. Source 95% intervals invert the hypergeometric distribution; combined-band bounds sum two 97.5% intervals. Cumulative bounds sum 40 stratum intervals at alpha=0.05/40 for conservative simultaneous coverage. Estimates and ordinary band intervals through K=1000 match the updated shorter analysis.</p>'
    doc+='<p>Labels measure broad investigative-topic relevance, including indirect and peripheral context, not wrongdoing, unique leads or corpus recall. Initial Codex screening is not human adjudicated and was not fully blinded to rank/source. Prior narrower judgments and existing exact-text judgments inform review; long messages use screening excerpts with targeted follow-up. Missing attachments are not reconstructed, though identified target documents can count as indirect leads. The same rubric applies throughout, and observed rebounds remain visible.</p>'
    doc+=semantic_reassessment_note(summary, SEMANTIC_REASSESSMENT_FILE)
    doc+=semantic_top200_note(summary, SEMANTIC_TOP200_REVIEW_FILE)
    for review_key, _, _ in REASSESSMENTS:
        if review_key not in summary:
            continue
        review = summary[review_key]
        doc+=(
            f'<p>The second review covered {review["sampled"]} sampled emails at combined ranks '
            f'{review["first_rank"]}–{review["last_rank"]}. '
            'Reviewers checked for transaction content or an identifiable record bearing on the investigation. '
            f'The relevant count changed from {review["relevant_before"]} to {review["relevant_after"]}. '
            f'Population-weighted precision across this range changed from {review["weighted_precision_before"]:.1%} '
            f'to {review["weighted_precision_after"]:.1%}. '
            f'<a href="{review_key}_changes.csv">See before-and-after verdicts and reasons.</a></p>'
        )
    if 'semantic_band_reassessment' in summary:
        review = summary['semantic_band_reassessment']
        doc+=(
            f'<p>A later review covered {review["sampled"]} semantic-attributed samples at combined ranks '
            '1101–1200 and 1401–1600. '
            f'The relevant count changed from {review["relevant_before"]} to {review["relevant_after"]}. '
            f'<a href="{SEMANTIC_BAND_REVIEW_FILE}">See changed verdicts, reasons and supporting records.</a></p>'
        )
    if 'band_1401_1500_reassessment' in summary:
        review = summary['band_1401_1500_reassessment']
        doc+=(
            f'<p>The most recent review checked all {review["sampled"]} previous positive labels in combined ranks '
            f'1401–1500 across both sources. {review["relevant_after"]} remained relevant. '
            f'<a href="{RANK_1500_REVIEW_FILE}">See the changed label and reason.</a></p>'
        )
    doc+=f'<p><b>Findings:</b> estimated broad-relevant emails: {c["estimated_relevant_through_1000"]:.1f} through K=1000 and {c["estimated_relevant_through_endpoint"]:.1f} through K=1920; the extension adds {c["estimated_relevant_through_endpoint"]-c["estimated_relevant_through_1000"]:.1f}. Original-horizon geometric elbow: K={original_elbow}; extended-horizon elbow: K={c["geometric_elbow_k"]}. The elbow maximizes normalized cumulative distance above the origin-to-endpoint chord. Changing the endpoint can change this descriptive budget candidate; it does not establish a unique optimal stopping K.</p>'
    doc+='<img src="precision_combined_and_sources_extended.png"><p><a href="sample_verdicts_compact.csv">780 verdicts and rationales</a> | <a href="precision_by_band_and_source.csv">Chart data</a> | <a href="combined_deduplicated.csv">1920 candidate ranking</a></p>'
    for source in ['combined','semantic','tfidf']:
        doc+=f'<h2>{source.title()}</h2>'+df[df.source.eq(source)][['band','population','sampled','relevant','precision','precision_ci_low','precision_ci_high','estimated_relevant','cumulative_estimated_relevant','cumulative_precision']].to_html(index=False,float_format=lambda x:f'{x:.3f}')
    doc+='<h2>Review audit</h2>'
    for r in reviewed.itertuples():
        doc+=f'<details><summary>{r.review_id} | rank {r.combined_rank} | {r.first_source} | relevant={r.relevant} | {r.tier} | {html.escape(r.subject)}</summary><p>{html.escape(r.rationale)}</p><p>Email ID: {r.email_id}; uncertainty: {r.uncertain}</p><pre>{html.escape(r.body_clean)}</pre></details>'
    (OUT/'analysis_report.html').write_text(doc+'</body></html>',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','read','evaluate','plot-combined']);p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=25);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='read':read(a.start,a.end)
    elif a.action == 'plot-combined':
        plot_combined_charts(dual_source_estimates())
    else:evaluate()
