#!/usr/bin/env python3
"""Build deployment-only partitions from finalized dashboard tables."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "outputs/grn_v1/dashboard_data"
DEST = ROOT / "data/dashboard_data"


def safe_name(value):
    return str(value).replace("/", "__")


def partition_csv(source_name, group_col, output_dir):
    frame = pd.read_csv(SOURCE / source_name, sep="\t", low_memory=False)
    target = DEST / output_dir
    target.mkdir(parents=True, exist_ok=True)
    for value, group in frame.groupby(group_col, sort=True):
        group.to_parquet(target / f"{safe_name(value)}.parquet", index=False)
    print(source_name, len(frame), "rows ->", frame[group_col].nunique(), "partitions")


def partition_parquet(source_name, group_col, output_dir):
    frame = pd.read_parquet(SOURCE / source_name)
    target = DEST / output_dir
    target.mkdir(parents=True, exist_ok=True)
    for value, group in frame.groupby(group_col, sort=True):
        group.to_parquet(target / f"{safe_name(value)}.parquet", index=False)
    print(source_name, len(frame), "rows ->", frame[group_col].nunique(), "partitions")


partition_csv("TF_centric_gene_edges.tsv.gz", "TF", "TF_centric_gene_edges_by_TF")
partition_csv("TF_centric_CRE_gene_edges.tsv.gz", "TF", "TF_centric_CRE_gene_edges_by_TF")
partition_parquet("archetype_structural_CRE_gene_edges.parquet", "frozen_archetype",
                  "archetype_structural_CRE_gene_edges_by_archetype")
partition_csv("condition_specific_pathway_enrichment.tsv.gz", "TF",
              "condition_specific_pathway_enrichment_by_TF")
