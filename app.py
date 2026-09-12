#!/usr/bin/env python3
"""SC.beta GRN TF-centric dashboard. Reads ONLY precomputed dashboard_data/ tables --
never re-runs GRN construction, TF disambiguation, or motif discovery. Read-if-exists
pattern throughout for MAGMA/S-LDSC so this app works before and after those jobs finish."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

CLUSTER_GRN = "/carter/users/juf009/projects/sc_beta_grn/outputs/grn_v1"
LOCAL_GRN = str(Path(__file__).resolve().parent / "data")
GRN = os.environ.get("SC_BETA_GRN_ROOT", CLUSTER_GRN if os.path.exists(CLUSTER_GRN) else LOCAL_GRN)
DD = f"{GRN}/dashboard_data"
CONDITIONS = ["control", "3-cyt", "IFNg", "dex", "palmitate", "Ex-4_HG"]
ASSOCIATION_CONDITIONS = ["3-cyt", "IFNg", "dex", "palmitate", "Ex-4_HG"]

st.set_page_config(page_title="SC.beta GRN -- TF-centric dashboard", layout="wide")

# ============================================================
# Data loading (cached, read-if-exists)
# ============================================================
@st.cache_data
def load_core():
    summary = pd.read_csv(f"{DD}/TF_centric_target_summary.tsv", sep="\t")
    gene_edges = pd.read_csv(f"{DD}/TF_centric_gene_edges.tsv.gz", sep="\t", low_memory=False)
    cre_edges = pd.read_csv(f"{DD}/TF_centric_CRE_gene_edges.tsv.gz", sep="\t", low_memory=False)
    struct = pd.read_parquet(f"{DD}/archetype_structural_CRE_gene_edges.parquet")
    cand_tf = pd.read_csv(f"{GRN}/tables/archetype_candidate_TFs.tsv", sep="\t")
    return summary, gene_edges, cre_edges, struct, cand_tf

@st.cache_data
def load_sequence_context():
    seq_ctx = pd.read_csv(f"{DD}/archetype_sequence_context.tsv", sep="\t")
    return seq_ctx

@st.cache_data
def load_condition_associations(tf):
    path = f"{DD}/condition_specific_TF_gene_associations/{tf}.tsv.gz"
    return pd.read_csv(path, sep="\t", low_memory=False)

@st.cache_data
def load_pathways():
    results_path = f"{DD}/condition_specific_pathway_enrichment.tsv.gz"
    summary_path = f"{DD}/condition_specific_pathway_enrichment_summary.tsv"
    if not os.path.exists(results_path) or not os.path.exists(summary_path):
        return None, None
    return (pd.read_csv(results_path, sep="\t", low_memory=False),
            pd.read_csv(summary_path, sep="\t", low_memory=False))

@st.cache_data
def load_finalized_disease_magma():
    signed_path = f"{GRN}/disease_genetics/magma_condition_associated_tf_signed/signed_condition_associated_tf_MAGMA_results.tsv"
    condition_supported = f"{GRN}/disease_genetics/magma_condition_supported"
    differential = f"{GRN}/disease_genetics/magma_differential_archetype"
    tf_results = pd.read_csv(signed_path, sep="\t") if os.path.exists(signed_path) else pd.DataFrame()
    archetype_paths = {
        "Structural baseline": f"{condition_supported}/MAGMA_structural_archetype_baseline_results.tsv",
        "Condition-supported": f"{condition_supported}/MAGMA_condition_supported_archetype_results.tsv",
        "Gained vs control": f"{differential}/MAGMA_gained_vs_control_results.tsv",
        "Perturbation-unique": f"{differential}/MAGMA_perturbation_unique_results.tsv",
    }
    archetype_results = {}
    for label, path in archetype_paths.items():
        archetype_results[label] = pd.read_csv(path, sep="\t") if os.path.exists(path) else pd.DataFrame()
    return tf_results, archetype_results

summary, gene_edges, cre_edges, struct, cand_tf = load_core()
seq_ctx = load_sequence_context()
pathways, pathway_summary = load_pathways()
tf_condition_magma, archetype_magma = load_finalized_disease_magma()

candidate_tfs = set(cand_tf[cand_tf["usable_for_candidate_TF_assignment"] == True]["candidate_TF"].dropna())
represented_archetypes = set(seq_ctx["frozen_archetype"].dropna())
represented_pairs = summary[summary["frozen_archetype"].isin(represented_archetypes)][["TF", "frozen_archetype"]].dropna().drop_duplicates()
usable_tf = sorted(candidate_tfs.intersection(represented_pairs["TF"]))
unmapped_tf = sorted(candidate_tfs.difference(usable_tf))

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("SC.beta GRN")
st.sidebar.caption("TF-centric view -- overlay on the frozen archetype -> CRE -> gene structural network")
if st.session_state.get("selected_tf") not in usable_tf:
    st.session_state["selected_tf"] = usable_tf[0]
tf_sel = st.sidebar.selectbox("Candidate TF", usable_tf, key="selected_tf")

tf_archs = sorted(represented_pairs[represented_pairs["TF"] == tf_sel]["frozen_archetype"].unique())
if st.session_state.get("selected_archetype") not in tf_archs:
    st.session_state["selected_archetype"] = tf_archs[0]
arch_sel = st.sidebar.selectbox("Compatible archetype", tf_archs, key="selected_archetype")
if unmapped_tf:
    st.sidebar.caption(f"{len(unmapped_tf):,} candidate TFs without a represented frozen archetype are hidden.")

tf_row = summary[(summary["TF"] == tf_sel) & (summary["frozen_archetype"] == arch_sel)]

st.title(f"TF: {tf_sel}  |  Archetype: {arch_sel}")

tabs = st.tabs(["Fixed Structural GRN", "Overview", "Target Genes", "Pathways", "Network", "Disease Genetics"])

# ============================================================
# FIXED STRUCTURAL GRN (condition-independent)
# ============================================================
with tabs[0]:
    st.header("Fixed Structural GRN")
    st.markdown("**Fixed structural scaffold: motif archetype → CRE → gene.**")
    st.caption("Frozen, condition-independent structure.")

    structural_edges = struct.drop_duplicates(subset=["frozen_archetype", "CRE_id", "gene"])
    global_counts = st.columns(5)
    global_counts[0].metric("Archetypes", f"{seq_ctx['frozen_archetype'].nunique():,}")
    global_counts[1].metric("Unique structural CREs", f"{structural_edges['CRE_id'].nunique():,}")
    global_counts[2].metric("Unique structural genes", f"{structural_edges['gene'].nunique():,}")
    global_counts[3].metric("Raw structural table rows", f"{len(struct):,}")
    global_counts[4].metric("Unique structural edges", f"{len(structural_edges):,}")
    st.caption("The frozen structural table has 1,208,323 raw rows. Deduplicating exact "
               "(frozen_archetype, CRE_id, gene) keys yields 878,197 unique structural edges; "
               "330,126 rows are repeated copies of an existing structural key.")

    st.subheader(f"Selected archetype: {arch_sel}")
    arch_context = seq_ctx[seq_ctx["frozen_archetype"] == arch_sel]
    arch_edges = structural_edges[structural_edges["frozen_archetype"] == arch_sel]
    if arch_context.empty:
        st.info("No frozen structural context is available for this archetype.")
    else:
        arch_context = arch_context.iloc[0]
        selected_counts = st.columns(4)
        selected_counts[0].metric("Static motif-positive CREs", f"{int(arch_context['n_static_motif_positive_CREs']):,}")
        selected_counts[1].metric("Structural GRN CREs", f"{arch_edges['CRE_id'].nunique():,}")
        selected_counts[2].metric("Unique structural edges", f"{len(arch_edges):,}")
        selected_counts[3].metric("Structural target genes", f"{arch_edges['gene'].nunique():,}")

        st.markdown("##### A. Selected-archetype structural funnel")
        funnel_labels = ["Static motif-positive CREs", "Structural CREs", "Structural target genes"]
        funnel_values = [int(arch_context["n_static_motif_positive_CREs"]),
                         arch_edges["CRE_id"].nunique(), arch_edges["gene"].nunique()]
        funnel_colors = ["#9ecae1", "#4292c6", "#08519c"]
        funnel_fig, funnel_ax = plt.subplots(figsize=(9, 3.2))
        max_value = max(funnel_values) if max(funnel_values) else 1
        for i, (label, value, color) in enumerate(zip(funnel_labels, funnel_values, funnel_colors)):
            width = value / max_value
            funnel_ax.barh(i, width, left=(1 - width) / 2, height=0.72, color=color)
            funnel_ax.text(0.5, i, f"{label}: {value:,}", ha="center", va="center",
                           color="white" if width > 0.35 else "black", fontsize=9, fontweight="bold")
        funnel_ax.set_xlim(0, 1)
        funnel_ax.set_ylim(-0.6, 2.6)
        funnel_ax.invert_yaxis()
        funnel_ax.axis("off")
        funnel_fig.patch.set_facecolor("white")
        st.pyplot(funnel_fig)

    archetype_overview = seq_ctx[["frozen_archetype", "n_static_motif_positive_CREs",
                                  "n_structural_GRN_CREs", "n_structural_target_genes"]].copy()
    archetype_overview = archetype_overview.sort_values("n_static_motif_positive_CREs", ascending=True)
    st.markdown("##### B. Global archetype overview")
    overview_fig, overview_ax = plt.subplots(figsize=(10, max(8, len(archetype_overview) * 0.24)))
    overview_plot = archetype_overview.set_index("frozen_archetype")
    overview_plot.plot.barh(ax=overview_ax, width=0.8,
                            color=["#9ecae1", "#4292c6", "#fdae6b"])
    overview_ax.set_xlabel("Count")
    overview_ax.set_ylabel("Motif archetype")
    overview_ax.legend(["Static CREs", "Structural CREs", "Target genes"], frameon=False)
    overview_ax.set_facecolor("white")
    overview_fig.patch.set_facecolor("white")
    for spine in overview_ax.spines.values():
        spine.set_visible(False)
    st.pyplot(overview_fig)

    edge_counts = (structural_edges.groupby("frozen_archetype").size()
                   .rename("n_structural_edges").reset_index())
    scatter_data = seq_ctx.merge(edge_counts, on="frozen_archetype", how="left")
    st.markdown("##### C. Structural CREs and target genes")
    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 5))
    point_sizes = 20 + 280 * scatter_data["n_structural_edges"] / scatter_data["n_structural_edges"].max()
    scatter_ax.scatter(scatter_data["n_structural_GRN_CREs"],
                       scatter_data["n_structural_target_genes"],
                       s=point_sizes, alpha=0.65, color="#3182bd", edgecolor="white", linewidth=0.6)
    selected_point = scatter_data[scatter_data["frozen_archetype"] == arch_sel]
    if not selected_point.empty:
        selected_point = selected_point.iloc[0]
        scatter_ax.scatter(selected_point["n_structural_GRN_CREs"],
                           selected_point["n_structural_target_genes"],
                           s=20 + 280 * selected_point["n_structural_edges"] / scatter_data["n_structural_edges"].max(),
                           color="#d62728", edgecolor="white", linewidth=0.8, zorder=3)
        scatter_ax.annotate(arch_sel, (selected_point["n_structural_GRN_CREs"],
                                      selected_point["n_structural_target_genes"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
    scatter_ax.set_xlabel("# structural CREs")
    scatter_ax.set_ylabel("# target genes")
    scatter_ax.set_title("Point size = # archetype→CRE→gene edges")
    scatter_ax.set_facecolor("white")
    scatter_fig.patch.set_facecolor("white")
    for spine in scatter_ax.spines.values():
        spine.set_visible(False)
    st.pyplot(scatter_fig)

    st.markdown("##### Structural targets")
    target_table = (arch_edges.groupby("gene", as_index=False)
                    .agg(n_linked_CREs=("CRE_id", "nunique"),
                         best_HC_link_score=("HC_link_score", "max"))
                    .sort_values(["n_linked_CREs", "best_HC_link_score", "gene"],
                                 ascending=[False, False, True]))
    st.dataframe(target_table, use_container_width=True, height=500)

# ============================================================
# TAB 1: Overview
# ============================================================
with tabs[1]:
    if len(tf_row) == 0:
        st.warning("No summary row for this TF x archetype combination.")
    else:
        r = tf_row.iloc[0]
        row1 = st.columns(3)
        row1[0].metric("TF", tf_sel)
        row1[1].metric("Compatible archetype", arch_sel)
        row1[2].metric("Motif family", r["motif_family"])
        row2 = st.columns(3)
        row2[0].metric("Baseline expression (logCPM)", f"{r['SCbeta_baseline_expression']:.2f}")
        row2[1].metric("TF-specific genes", f"{int(r['n_TF_specific_targets']):,}")
        row2[2].metric("Top condition", r["archetype_top_model_support_condition"])
        st.caption("Additional structural targets may be shared across compatible TFs or remain unresolved at the exact-TF level.")

# ============================================================
# TAB 2: Target Genes
# ============================================================
with tabs[2]:
    st.subheader("Condition-specific TF–gene associations")
    st.caption("These associations are computed independently within each condition using TF and gene response "
               "profiles. Empirical P values are derived from the condition-specific matched-null analysis. "
               "Correlation does not demonstrate direct TF binding or causal regulation.")
    st.caption("This exploratory section tests associated genes across the frozen target-gene universe; "
               "the compatible-archetype selector does not restrict these condition-specific associations.")

    associations = load_condition_associations(tf_sel)

    summary_columns = st.columns(len(ASSOCIATION_CONDITIONS))
    for column, condition in zip(summary_columns, ASSOCIATION_CONDITIONS):
        condition_data = associations[associations["condition"] == condition]
        with column:
            with st.container(border=True):
                st.markdown(f"**{condition}**")
                st.metric("Tested genes", f"{condition_data['gene'].nunique():,}")
                st.metric("Associated genes (P<0.01)",
                          f"{condition_data.loc[condition_data['empirical_p_condition'] < 0.01, 'gene'].nunique():,}")
                st.metric("Median |rho|", f"{condition_data['rho_condition'].abs().median():.3f}")

    st.markdown("##### Gene × condition correlation heatmap")
    n_heatmap_genes = st.slider("Genes shown", min_value=30, max_value=200, value=40, step=10,
                                key="condition_association_heatmap_n")
    gene_rank = (associations.groupby("gene")
                 .agg(min_empirical_p=("empirical_p_condition", "min"),
                      max_abs_rho=("rho_condition", lambda values: values.abs().max()))
                 .sort_values(["min_empirical_p", "max_abs_rho"], ascending=[True, False]))
    heatmap_genes = gene_rank.head(n_heatmap_genes).index.tolist()
    heatmap_data = associations[associations["gene"].isin(heatmap_genes)]
    rho_matrix = (heatmap_data.pivot(index="gene", columns="condition", values="rho_condition")
                  .reindex(index=heatmap_genes, columns=ASSOCIATION_CONDITIONS))
    p_matrix = (heatmap_data.pivot(index="gene", columns="condition", values="empirical_p_condition")
                .reindex(index=heatmap_genes, columns=ASSOCIATION_CONDITIONS))

    heatmap_fig, heatmap_ax = plt.subplots(figsize=(8, max(6, n_heatmap_genes * 0.24)))
    heatmap_cmap = plt.cm.RdBu_r.copy()
    heatmap_cmap.set_bad("#d9d9d9")
    image = heatmap_ax.imshow(np.ma.masked_invalid(rho_matrix.to_numpy(float)), aspect="auto",
                              interpolation="nearest", cmap=heatmap_cmap, vmin=-1, vmax=1)
    heatmap_ax.set_xticks(range(len(ASSOCIATION_CONDITIONS)))
    heatmap_ax.set_xticklabels(ASSOCIATION_CONDITIONS, rotation=30, ha="right")
    heatmap_ax.set_yticks(range(len(heatmap_genes)))
    heatmap_ax.set_yticklabels(heatmap_genes, fontsize=7)
    for row_i in range(len(heatmap_genes)):
        for col_i in range(len(ASSOCIATION_CONDITIONS)):
            p_value = p_matrix.iat[row_i, col_i]
            if pd.notna(p_value) and p_value < 0.01:
                stars = "**" if p_value < 0.001 else "*"
                heatmap_ax.text(col_i, row_i, stars, ha="center", va="center",
                                fontsize=7, fontweight="bold", color="black")
    heatmap_ax.set_xlabel("Condition")
    heatmap_ax.set_ylabel("Associated gene")
    heatmap_fig.colorbar(image, ax=heatmap_ax, label="Spearman rho", shrink=0.7)
    heatmap_fig.patch.set_facecolor("white")
    st.pyplot(heatmap_fig)
    plt.close(heatmap_fig)
    st.caption("Color represents condition-specific Spearman correlation between TF and gene response profiles. "
               "Stars indicate matched-null empirical P values. * P<0.01; ** P<0.001. Gray/blank indicates "
               "insufficient observations or missing values.")

    st.markdown("##### Correlation distributions")
    distribution_fig, distribution_ax = plt.subplots(figsize=(9, 4.2))
    distribution_values = [associations.loc[associations["condition"] == condition,
                                            "rho_condition"].dropna().to_numpy()
                           for condition in ASSOCIATION_CONDITIONS]
    violin = distribution_ax.violinplot(distribution_values, showmeans=False, showmedians=True,
                                        showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#4292c6")
        body.set_edgecolor("#2171b5")
        body.set_alpha(0.65)
    violin["cmedians"].set_color("#08306b")
    distribution_ax.axhline(0, color="#777777", linewidth=0.8)
    distribution_ax.set_xticks(range(1, len(ASSOCIATION_CONDITIONS) + 1))
    distribution_ax.set_xticklabels(ASSOCIATION_CONDITIONS, rotation=25, ha="right")
    distribution_ax.set_ylabel("Condition-specific Spearman rho")
    distribution_ax.set_ylim(-1.05, 1.05)
    distribution_ax.set_facecolor("white")
    distribution_fig.patch.set_facecolor("white")
    for spine in distribution_ax.spines.values():
        spine.set_visible(False)
    st.pyplot(distribution_fig)
    plt.close(distribution_fig)

    st.markdown("##### Detailed associations")
    filter_columns = st.columns(4)
    with filter_columns[0]:
        association_condition_filter = st.multiselect(
            "Condition", ASSOCIATION_CONDITIONS, default=ASSOCIATION_CONDITIONS,
            key="condition_association_conditions")
    with filter_columns[1]:
        association_p_filter = st.number_input("Maximum empirical P", min_value=0.0, max_value=1.0,
                                                value=0.01, step=0.001, format="%.3f",
                                                key="condition_association_p")
    with filter_columns[2]:
        association_rho_filter = st.slider("Minimum |rho|", min_value=0.0, max_value=1.0,
                                            value=0.0, step=0.05, key="condition_association_rho")
    with filter_columns[3]:
        association_gene_search = st.text_input("Gene search", key="condition_association_gene")

    association_table = associations[
        associations["condition"].isin(association_condition_filter) &
        (associations["empirical_p_condition"] <= association_p_filter) &
        (associations["rho_condition"].abs() >= association_rho_filter)
    ].copy()
    if association_gene_search.strip():
        association_table = association_table[
            association_table["gene"].str.contains(association_gene_search.strip(), case=False, regex=False)]
    association_table = association_table.rename(columns={"rho_global_reference": "global_rho (reference only)"})
    association_table = association_table.sort_values(
        ["empirical_p_condition", "rho_condition"], ascending=[True, False])
    association_columns = ["gene", "condition", "n_observations", "rho_condition",
                           "empirical_p_condition", "global_rho (reference only)"]
    st.caption(f"{len(association_table):,} associated gene–condition rows match the current filters.")
    st.dataframe(association_table[association_columns], use_container_width=True, height=500)

# ============================================================
# TAB 3: Condition-specific pathways
# ============================================================
with tabs[3]:
    st.header("Condition-specific pathway enrichment")
    with st.container(border=True):
        st.markdown("**How these gene sets are built**")
        st.write("For each condition, we compare the selected TF's RNA-response pattern with every gene in "
                 "the frozen GRN target universe. Genes with matched-null empirical P < 0.01 form the "
                 "condition-associated response gene set. We then test whether curated biological pathways "
                 "contain more of those genes than expected by chance.")
        st.caption("This is pathway enrichment of condition-associated TF–gene correlations. It does not "
                   "demonstrate direct TF binding or causal regulation.")

    if pathways is None or pathway_summary is None:
        st.warning("Condition-specific pathway enrichment results are not available.")
    else:
        controls = st.columns(2)
        with controls[0]:
            pathway_condition = st.selectbox("Condition", ASSOCIATION_CONDITIONS,
                                             key="pathway_condition")
        with controls[1]:
            database_order = [database for database in ["Hallmark", "Reactome", "GO_BP"]
                              if database in set(pathway_summary["database"])]
            db_sel = st.selectbox("Pathway database", database_order, key="pathway_database")

        summary_here = pathway_summary[(pathway_summary["TF"] == tf_sel) &
                                       (pathway_summary["database"] == db_sel)].copy()
        summary_here["condition"] = pd.Categorical(summary_here["condition"], ASSOCIATION_CONDITIONS,
                                                    ordered=True)
        summary_here = summary_here.sort_values("condition")
        selected_summary = summary_here[summary_here["condition"] == pathway_condition]
        if selected_summary.empty:
            st.info(f"No pathway-enrichment summary is available for {tf_sel} in {pathway_condition}.")
        else:
            selected_summary = selected_summary.iloc[0]
            cards = st.columns(4)
            cards[0].metric("Associated genes", f"{int(selected_summary['n_query_genes']):,}")
            cards[1].metric("Positive / negative rho",
                            f"{int(selected_summary['n_positive_rho_genes']):,} / "
                            f"{int(selected_summary['n_negative_rho_genes']):,}")
            cards[2].metric("Pathways tested", f"{int(selected_summary['n_pathways_tested']):,}")
            cards[3].metric("Pathways at FDR < 0.05",
                            f"{int(selected_summary['n_pathways_FDR_lt_0.05']):,}")

            st.markdown("##### Condition overview")
            overview = summary_here.rename(columns={
                "n_query_genes": "associated genes",
                "n_pathways_FDR_lt_0.05": "pathways at FDR < 0.05",
                "minimum_P": "minimum P", "minimum_BH_FDR": "minimum BH-FDR",
                "query_status": "status",
            })
            st.dataframe(overview[["condition", "associated genes", "pathways at FDR < 0.05",
                                   "minimum P", "minimum BH-FDR", "status"]],
                         use_container_width=True, hide_index=True)

            pw_sub = pathways[(pathways["database"] == db_sel) &
                              (pathways["TF"] == tf_sel) &
                              (pathways["condition"] == pathway_condition)].copy()
            if selected_summary["query_status"] != "testable":
                st.info("This condition has fewer than 10 associated genes, so pathway enrichment was not tested.")
            elif pw_sub.empty:
                st.info("No pathway overlaps were found for this TF, condition, and database.")
            else:
                pw_sub = pw_sub.sort_values(["BH_FDR", "P"]).head(20)
                st.markdown(f"##### Top pathways · {pathway_condition}")
                st.dataframe(pw_sub[["pathway", "n_query_genes", "n_pathway_genes", "n_overlap",
                                     "odds_ratio", "P", "BH_FDR", "overlap_genes"]],
                             use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(8, max(3, len(pw_sub) * 0.35)))
                neglogfdr = -np.log10(pw_sub["BH_FDR"].clip(lower=1e-300))
                sizes = pw_sub["n_overlap"] * 20
                sc = ax.scatter(pw_sub["odds_ratio"], range(len(pw_sub)), s=sizes, c=neglogfdr, cmap="viridis")
                ax.set_yticks(range(len(pw_sub))); ax.set_yticklabels(pw_sub["pathway"], fontsize=7)
                ax.invert_yaxis()
                ax.set_xlabel("Odds ratio")
                plt.colorbar(sc, ax=ax, label="-log10(BH FDR)")
                ax.set_facecolor("white"); fig.patch.set_facecolor("white")
                for s in ax.spines.values(): s.set_visible(False)
                st.pyplot(fig)
                plt.close(fig)
                if (pw_sub["BH_FDR"] >= 0.05).all():
                    st.caption("No displayed pathway reaches BH-FDR < 0.05; these rows are exploratory.")
                st.caption("One-sided hypergeometric over-representation analysis. BH-FDR is corrected within "
                           "this TF × condition × database across all tested pathways, including pathways with "
                           "zero overlap.")

# ============================================================
# TAB 4: Network
# ============================================================
with tabs[4]:
    top_n = st.slider("Top N CREs / genes to display", 5, 20, 10)
    sub = gene_edges[(gene_edges["TF"] == tf_sel) & (gene_edges["frozen_archetype"] == arch_sel) &
                      (gene_edges["resolution"] == "TF_specific")].copy()
    if len(sub) == 0:
        sub = gene_edges[(gene_edges["TF"] == tf_sel) & (gene_edges["frozen_archetype"] == arch_sel)].copy()
        st.info("No TF-specific targets -- showing best-available (regulator-group/unresolved) targets instead.")
    sub = sub.sort_values("empirical_p", na_position="last").head(top_n)
    cre_sub = cre_edges[(cre_edges["TF"] == tf_sel) & (cre_edges["frozen_archetype"] == arch_sel) &
                         (cre_edges["gene"].isin(sub["gene"]))].drop_duplicates(subset=["CRE_id", "gene"]).head(top_n * 2)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.4)))
    ax.axis("off")
    ax.text(0.05, 0.5, tf_sel, fontsize=13, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#e8d5f0"))
    ax.text(0.25, 0.5, arch_sel, fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#d6e9f8"))
    ax.annotate("", xy=(0.20, 0.5), xytext=(0.10, 0.5), arrowprops=dict(arrowstyle="->"))
    y_positions = np.linspace(0.05, 0.95, max(len(cre_sub), 1))
    for i, (_, row) in enumerate(cre_sub.iterrows()):
        y = y_positions[i]
        ax.text(0.5, y, row["CRE_id"], fontsize=6.5, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="#d0f0d0"))
        ax.annotate("", xy=(0.45, y), xytext=(0.30, 0.5), arrowprops=dict(arrowstyle="->", lw=0.6, alpha=0.5))
        ax.text(0.85, y, row["gene"], fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="#fbe5c8"))
        ax.annotate("", xy=(0.80, y), xytext=(0.55, y), arrowprops=dict(arrowstyle="->", lw=0.6, alpha=0.5))
    fig.patch.set_facecolor("white")
    st.pyplot(fig)
    st.caption("Small worked-example subnetwork only -- not the full 1.2M-edge structural GRN.")

# ============================================================
# TAB 5: Disease Genetics
# ============================================================
with tabs[5]:
    gwas_order = ["T1D", "T2D_suzuki2024", "T2D_diamante2022"]
    gwas_labels = {"T1D": "T1D", "T2D_suzuki2024": "Suzuki T2D",
                   "T2D_diamante2022": "DIAMANTE T2D",
                   "T2D Suzuki 2024": "Suzuki T2D",
                   "T2D DIAMANTE 2022": "DIAMANTE T2D"}
    archetype_gwas_order = ["T1D", "T2D Suzuki 2024", "T2D DIAMANTE 2022"]

    def magma_heatmap(frame, row_col, row_order, title, q_within_col, q_global_col):
        beta_col = "beta" if "beta" in frame.columns else "BETA"
        matrix = frame.pivot_table(index=row_col, columns="GWAS", values=beta_col, aggfunc="first")
        matrix = matrix.reindex(index=row_order, columns=archetype_gwas_order if "T2D Suzuki 2024" in frame["GWAS"].unique() else gwas_order)
        values = matrix.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
        limit = max(limit, 0.01)
        fig, ax = plt.subplots(figsize=(7.3, max(2.2, 0.44 * len(matrix) + 1.1)))
        image = ax.imshow(masked, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_facecolor("#d9d9d9")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([gwas_labels.get(x, x) for x in matrix.columns], fontsize=9)
        ax.set_yticks(range(len(matrix.index))); ax.set_yticklabels(matrix.index, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        for row_i, row_name in enumerate(matrix.index):
            for col_i, gwas in enumerate(matrix.columns):
                hit = frame[(frame[row_col] == row_name) & (frame["GWAS"] == gwas)]
                if hit.empty or pd.isna(matrix.loc[row_name, gwas]):
                    continue
                hit = hit.iloc[0]
                mark = "**" if q_global_col and pd.notna(hit.get(q_global_col)) and hit[q_global_col] < 0.05 else (
                    "*" if q_within_col and pd.notna(hit.get(q_within_col)) and hit[q_within_col] < 0.05 else "")
                ax.text(col_i, row_i, f"{matrix.loc[row_name, gwas]:.2f}{mark}", ha="center", va="center",
                        fontsize=8, color="white" if abs(matrix.loc[row_name, gwas]) > limit * 0.55 else "#222")
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.colorbar(image, ax=ax, shrink=0.72, label="MAGMA beta")
        fig.patch.set_facecolor("white")
        return fig

    st.header("1. TF-level condition-associated MAGMA")
    with st.container(border=True):
        st.markdown("**How this gene set is built**")
        st.write("Within each condition, we compare the selected TF's RNA-response pattern with every "
                 "structural GRN gene's response pattern across the available contrasts. Genes whose "
                 "TF–gene correlation passes the matched-null empirical P < 0.01 threshold form that "
                 "condition's response gene set. MAGMA then tests whether the set is enriched for "
                 "T1D or T2D genetic association.")

    tf_primary = tf_condition_magma[
        (tf_condition_magma["TF"] == tf_sel) &
        (tf_condition_magma["analysis_class"] == "unsigned_sensitivity")
    ].copy()
    if tf_primary.empty:
        st.info(f"No finalized TF-level MAGMA results are available for {tf_sel}.")
    else:
        tf_primary["T2D replicated"] = False
        for condition in ASSOCIATION_CONDITIONS:
            t2d = tf_primary[(tf_primary["condition"] == condition) &
                             (tf_primary["GWAS"].isin(["T2D_suzuki2024", "T2D_diamante2022"]))]
            replicated = len(t2d) == 2 and (t2d["Q_BH_within_condition_class"] < 0.05).all()
            tf_primary.loc[tf_primary["condition"] == condition, "T2D replicated"] = replicated

        tested_conditions = tf_primary["condition"].nunique()
        testable_sets = tf_primary.groupby("condition")["set_id"].nunique().size
        within_hits = int((tf_primary["Q_BH_within_condition_class"] < 0.05).sum())
        replicated_conditions = tf_primary.loc[tf_primary["T2D replicated"], "condition"].nunique()
        metrics = st.columns(4)
        metrics[0].metric("Tested conditions", tested_conditions)
        metrics[1].metric("MAGMA-testable condition sets", testable_sets)
        metrics[2].metric("Within-condition FDR signals", within_hits)
        metrics[3].metric("T2D-replicated conditions", replicated_conditions)
        if replicated_conditions:
            labels = ", ".join(tf_primary.loc[tf_primary["T2D replicated"], "condition"].drop_duplicates())
            st.success(f"T2D replicated · {labels}")

        st.subheader(f"Condition-associated TF response gene set: {tf_sel}")
        st.pyplot(magma_heatmap(tf_primary, "condition", ASSOCIATION_CONDITIONS,
                                "All associated genes, regardless of correlation direction",
                                "Q_BH_within_condition_class",
                                "Q_BH_across_analysis_class"), use_container_width=False)
        st.caption("Color and cell value show MAGMA beta. * within-condition BH-FDR < 0.05; "
                   "** global TF × condition BH-FDR < 0.05; gray = untested or insufficient.")
        if not (tf_primary["Q_BH_across_analysis_class"] < 0.05).any():
            st.info("No result for this TF survives global TF × condition correction.")

        display = tf_primary.copy()
        display["GWAS"] = display["GWAS"].map(gwas_labels)
        display["GWAS"] = pd.Categorical(display["GWAS"],
                                          ["T1D", "Suzuki T2D", "DIAMANTE T2D"], ordered=True)
        display = display.rename(columns={
            "NGENES_MAGMA": "mapped gene count", "BETA": "MAGMA beta", "SE": "SE",
            "P": "P", "Q_BH_within_condition_class": "within-condition Q",
            "Q_BH_across_analysis_class": "global Q",
        })
        display["condition"] = pd.Categorical(display["condition"], ASSOCIATION_CONDITIONS, ordered=True)
        display = display.sort_values(["condition", "GWAS"])
        st.dataframe(display[["condition", "GWAS", "mapped gene count", "MAGMA beta", "SE", "P",
                              "within-condition Q", "global Q", "T2D replicated"]],
                     use_container_width=True, hide_index=True)

        st.warning("These TF-level gene sets are exploratory. They are defined from per-condition TF–gene "
                   "correlations using only 7–8 observations. Within-condition FDR signals do not imply "
                   "study-wide significance or direct TF regulation.")

        with st.expander("Directionality sensitivity analysis: positive-rho and negative-rho sets"):
            st.caption("This separates association direction for interpretation. It is not a solution to the "
                       "small-sample instability problem.")
            for direction, label in [("positive", "Positive-rho"), ("negative", "Negative-rho")]:
                directed = tf_condition_magma[(tf_condition_magma["TF"] == tf_sel) &
                                               (tf_condition_magma["analysis_class"] == direction)].copy()
                st.markdown(f"**{label} response program**")
                if directed.empty:
                    st.info("No MAGMA-testable sets are available.")
                    continue
                st.pyplot(magma_heatmap(directed, "condition", ASSOCIATION_CONDITIONS, label,
                                        "Q_BH_within_condition_class", "Q_BH_across_primary_signed"),
                          use_container_width=False)
                directed["GWAS"] = directed["GWAS"].map(gwas_labels)
                st.dataframe(directed.rename(columns={"NGENES_MAGMA": "mapped gene count", "BETA": "MAGMA beta",
                                                       "Q_BH_within_condition_class": "within-condition Q",
                                                       "Q_BH_across_primary_signed": "global signed Q"})[
                    ["condition", "GWAS", "mapped gene count", "MAGMA beta", "SE", "P",
                     "within-condition Q", "global signed Q"]], use_container_width=True, hide_index=True)

    st.divider()
    st.header("2. Archetype-level regulatory genetics")
    with st.container(border=True):
        st.markdown("**How these gene sets are built**")
        st.write("We begin with the selected motif archetype, find its CREs in the frozen structural GRN, "
                 "and follow the fixed CRE→gene links. The rows apply increasingly condition-aware "
                 "sequence-model filters to that same archetype-level scaffold before MAGMA tests the linked genes.")
        st.markdown(
            "- **Structural baseline:** all genes linked to the archetype through the frozen CRE→gene scaffold; "
            "this reference set is the same in every condition.\n"
            "- **Condition-supported:** genes linked through structural CREs where the sequence model detects "
            "that archetype in the indicated condition.\n"
            "- **Gained vs control:** genes linked through CREs that gain archetype-level sequence-model support "
            "in the perturbation compared with control."
        )
        st.caption("These are motif-archetype gene sets, so enrichment cannot be assigned to one exact TF.")

    archetype_parts = []
    for evidence_label, frame in archetype_magma.items():
        if evidence_label == "Perturbation-unique":
            continue
        if frame.empty:
            continue
        part = frame[frame["frozen_archetype"] == arch_sel].copy()
        if part.empty:
            continue
        part["evidence level"] = evidence_label
        part["display row"] = np.where(part["condition"].fillna("") == "", evidence_label,
                                        evidence_label + " · " + part["condition"].fillna(""))
        part["within-analysis Q"] = part["BH_FDR_within_analysis_class"] if "BH_FDR_within_analysis_class" in part else part["BH_FDR"]
        archetype_parts.append(part)
    if not archetype_parts:
        st.info(f"No finalized archetype-level MAGMA results are available for {arch_sel}.")
    else:
        archetype_view = pd.concat(archetype_parts, ignore_index=True)
        evidence_order = ["Structural baseline"] + [
            f"{label} · {condition}" for label in ["Condition-supported", "Gained vs control"]
            for condition in ASSOCIATION_CONDITIONS]
        present_order = [row for row in evidence_order if row in set(archetype_view["display row"])]
        st.subheader(f"Archetype: {arch_sel}")
        st.pyplot(magma_heatmap(archetype_view, "display row", present_order,
                                "Archetype-level MAGMA", "within-analysis Q", None),
                  use_container_width=False)
        st.caption("Color and cell value show MAGMA beta. * BH-FDR < 0.05 within the corresponding "
                   "archetype analysis class; gray = untested or insufficient.")
        archetype_table = archetype_view.copy()
        archetype_table["GWAS"] = archetype_table["GWAS"].map(gwas_labels)
        archetype_table["GWAS"] = pd.Categorical(archetype_table["GWAS"],
                                                  ["T1D", "Suzuki T2D", "DIAMANTE T2D"], ordered=True)
        archetype_table = archetype_table.rename(columns={"n_genes_tested": "mapped gene count",
                                                           "beta": "MAGMA beta"})
        archetype_table = archetype_table.sort_values(["evidence level", "condition", "GWAS"])
        st.dataframe(archetype_table[["evidence level", "condition", "GWAS", "mapped gene count",
                                      "MAGMA beta", "SE", "P", "within-analysis Q", "status"]],
                     use_container_width=True, hide_index=True)
