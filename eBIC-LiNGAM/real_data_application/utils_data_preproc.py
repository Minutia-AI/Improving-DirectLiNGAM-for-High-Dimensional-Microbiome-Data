from __future__ import annotations

from typing import Optional, Literal
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


@dataclass
class CombinedInputs:
    rel_abund: pd.DataFrame
    metadata: pd.DataFrame


def create_combined_rel_abund_and_metadata(
    train_rel_path: str,
    test_rel_path: str,
    train_meta_path: str,
    test_meta_path: str,
    *,
    taxa_col: str = "Unnamed: 0",        # in your files this is the taxonomy column
    sample_id_col: str = "sample_id",    # in your metadata
    add_split_column: bool = True,
    split_colname: str = "split",
    train_label: str = "train",
    test_label: str = "test",
    fill_missing_rel_abund_with_zero: bool = True,
    out_rel_path: Optional[str] = None,
    out_meta_path: Optional[str] = None,
) -> CombinedInputs:
    """
    Create:
      1) One combined relative-abundance table (taxa x samples) by joining train+test on taxa.
      2) One combined metadata table (samples x fields) by stacking train+test rows.

    This function does NOT apply taxonomy-level preprocessing yet.
    You can run your preprocessing afterwards on the combined relative abundances,
    then merge the preprocessed taxa table with the combined metadata.

    Parameters
    ----------
    taxa_col:
        Column in the rel-abund CSV containing the taxa/taxonomy string.
        (Your files use 'Unnamed: 0'.)
    sample_id_col:
        Column in metadata containing sample IDs.
    add_split_column:
        If True, adds a column (default 'split') = 'train' or 'test' to metadata.
    fill_missing_rel_abund_with_zero:
        If True, taxa that appear only in one split get 0 in the other split samples.
    out_rel_path/out_meta_path:
        If provided, saves the combined tables as CSV.

    Returns
    -------
    CombinedInputs(rel_abund, metadata)
        rel_abund: DataFrame with taxa as first column and sample columns after.
        metadata: DataFrame with sample rows (and optional split column).
    """

    # -------------------------
    # Load relative abundances
    # -------------------------
    rel_train = pd.read_csv(train_rel_path)
    rel_test = pd.read_csv(test_rel_path)

    # Validate taxa column presence
    for name, df in [("train_rel", rel_train), ("test_rel", rel_test)]:
        if taxa_col not in df.columns:
            raise ValueError(
                f"{name} missing taxa_col='{taxa_col}'. "
                f"Found columns: {list(df.columns)[:10]} ..."
            )

    # Set taxa as index for safe alignment; ensure duplicates are handled
    rel_train = rel_train.set_index(taxa_col)
    rel_test = rel_test.set_index(taxa_col)

    # If taxa index has duplicates, aggregate by sum (common in some exports)
    if not rel_train.index.is_unique:
        rel_train = rel_train.groupby(level=0).sum()
    if not rel_test.index.is_unique:
        rel_test = rel_test.groupby(level=0).sum()

    # Check for overlapping sample columns (should usually be disjoint)
    overlap_samples = set(rel_train.columns).intersection(set(rel_test.columns))
    if overlap_samples:
        raise ValueError(
            f"Train/Test relative abundance files share sample columns (unexpected): "
            f"{sorted(list(overlap_samples))[:10]} ..."
        )

    # Join columns by taxa (outer keeps union of taxa)
    rel_all = rel_train.join(rel_test, how="outer")

    # Convert to numeric, coerce junk to NaN then optionally fill with 0
    rel_all = rel_all.apply(pd.to_numeric, errors="coerce")
    if fill_missing_rel_abund_with_zero:
        rel_all = rel_all.fillna(0)

    # Put taxa back as a normal column (same format as your original files)
    rel_all = rel_all.reset_index().rename(columns={taxa_col: taxa_col})

    # -------------------------
    # Load metadata
    # -------------------------
    meta_train = pd.read_csv(train_meta_path)
    meta_test = pd.read_csv(test_meta_path)

    for name, df in [("train_meta", meta_train), ("test_meta", meta_test)]:
        if sample_id_col not in df.columns:
            raise ValueError(
                f"{name} missing sample_id_col='{sample_id_col}'. "
                f"Found columns: {list(df.columns)}"
            )

    if add_split_column:
        meta_train[split_colname] = train_label
        meta_test[split_colname] = test_label

    # Stack rows (union of columns, keep whatever exists in either split)
    meta_all = pd.concat([meta_train, meta_test], axis=0, ignore_index=True, sort=False)

    # Basic sanity check: duplicated sample IDs in metadata
    dup = meta_all[sample_id_col].duplicated()
    if dup.any():
        duplicated_ids = meta_all.loc[dup, sample_id_col].astype(str).unique().tolist()
        raise ValueError(
            f"Duplicate sample_id values found in combined metadata: "
            f"{duplicated_ids[:10]} ..."
        )

    # -------------------------
    # Optional: save outputs
    # -------------------------
    if out_rel_path is not None:
        rel_all.to_csv(out_rel_path, index=False)
    if out_meta_path is not None:
        meta_all.to_csv(out_meta_path, index=False)

    return CombinedInputs(rel_abund=rel_all, metadata=meta_all)


def extract_taxonomy_level(
    taxon: str,
    level: int,
    sep: str = "|",
    *,
    strip_prefix: bool = True,
    prefixes: tuple[str, ...] = ("k__", "p__", "c__", "o__", "f__", "g__", "s__"),
    missing: str = "Unassigned",
) -> str:
    """
    Extract a taxonomy level from a pipe-separated taxonomy string.

    Example taxonomy string:
        "k__Bacteria|p__Firmicutes|c__Clostridia|o__..."

    Parameters
    ----------
    taxon : str
        The taxonomy string.
    level : int
        1-based level to extract (1..N). If out of range, returns `missing`.
    sep : str
        Separator between levels (default '|').
    strip_prefix : bool
        If True, removes prefixes like 'g__' and 's__'.
    prefixes : tuple[str, ...]
        Prefixes to remove when `strip_prefix=True`.
    missing : str
        Returned when level is invalid or missing.

    Returns
    -------
    str
        Extracted taxonomy label at the requested level.
    """
    if not isinstance(taxon, str) or not taxon.strip():
        return missing
    if level < 1:
        return missing

    parts = [p.strip() for p in taxon.split(sep)]
    if level > len(parts):
        return missing

    label = parts[level - 1].strip()
    if strip_prefix:
        for pref in prefixes:
            if label.startswith(pref):
                label = label[len(pref):]
                break

    # Normalize empty/unknown labels
    if not label or label in {"__", "NA", "nan"}:
        return missing

    return label


def preprocess_otu_dataset(
    path_otu: str,
    level: int,
    *,
    taxonomy_col: Optional[str] = None,
    sep: str = "|",
    drop_all_zero_samples: bool = True,
    drop_all_zero_taxa: bool = True,
    dtype: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and preprocess an OTU/taxa table.

    Expected input shape:
        rows   = taxa (taxonomy strings)
        cols   = samples (counts/abundances)
        plus one taxonomy column (e.g., 'Taxa' or first column)

    Output shape:
        index  = sample_id
        cols   = aggregated taxa at desired taxonomy level
        values = numeric counts/abundances

    Notes
    -----
    - Removes all-zero samples (columns) and all-zero taxa (after transposition),
      if enabled.
    - Aggregates taxa by the extracted taxonomy level via sum.
    """
    # Read file
    df = pd.read_csv(path_otu, dtype=dtype)

    # Identify taxonomy column robustly
    if taxonomy_col is None:
        candidate_cols = ["Taxa", "taxonomy", "Taxonomy", "Unnamed: 0"]
        found = next((c for c in candidate_cols if c in df.columns), None)
        taxonomy_col = found if found is not None else df.columns[0]

    # Rename to a standard name
    if taxonomy_col != "Taxa":
        df = df.rename(columns={taxonomy_col: "Taxa"})
    else:
        # ensure exact casing
        df = df.rename(columns={"Taxa": "Taxa"})

    # Coerce sample columns to numeric (taxonomy stays as object)
    sample_cols = [c for c in df.columns if c != "Taxa"]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce")

    # Treat NaN as 0 for count tables
    df[sample_cols] = df[sample_cols].fillna(0)

    # Optionally drop all-zero sample columns (before aggregation)
    if drop_all_zero_samples and sample_cols:
        nonzero_sample_cols = [c for c in sample_cols if (df[c] != 0).any()]
        df = df[["Taxa"] + nonzero_sample_cols]
        sample_cols = nonzero_sample_cols  # update

    # Extract desired taxonomy level
    df["Taxa"] = df["Taxa"].apply(lambda x: extract_taxonomy_level(x, level, sep=sep))

    # Drop unassigned/empty taxa rows if you don't want them
    df = df[df["Taxa"].notna() & (df["Taxa"].astype(str).str.strip() != "")]
    # If you want to explicitly remove 'Unassigned', uncomment:
    # df = df[df["Taxa"] != "Unassigned"]

    # Aggregate by taxa name
    df = df.groupby("Taxa", as_index=False)[sample_cols].sum()

    # Transpose: samples become rows
    df = df.set_index("Taxa").T
    df.index.name = "sample_id"

    # Optionally drop taxa columns that are all-zero after transpose
    if drop_all_zero_taxa and len(df.columns) > 0:
        df = df.loc[:, (df != 0).any(axis=0)]

    return df


def preprocess_metadata_dataset(
    path_metadata: str,
    *,
    sample_id_col: str = "sample_id",
    dtype: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load metadata and set sample_id as index.
    """
    meta = pd.read_csv(path_metadata, dtype=dtype)
    if sample_id_col not in meta.columns:
        raise ValueError(
            f"Metadata file must contain a '{sample_id_col}' column. "
            f"Found columns: {list(meta.columns)}"
        )
    meta = meta.set_index(sample_id_col)
    return meta


def merge_datasets(
    path_otu: str,
    path_metadata: str,
    level: int,
    *,
    taxonomy_col: Optional[str] = None,
    sep: str = "|",
    how: Literal["inner", "left", "right", "outer"] = "inner",
    validate_unique_sample_ids: bool = True,
) -> pd.DataFrame:
    """
    Preprocess OTU and metadata, then merge on sample_id.

    Parameters
    ----------
    how : {'inner','left','right','outer'}
        Join strategy:
        - inner: keep only sample_ids present in both
        - left: keep all metadata rows
        - right: keep all taxa rows
        - outer: union of sample_ids
    """
    taxa = preprocess_otu_dataset(path_otu, level, taxonomy_col=taxonomy_col, sep=sep)
    meta = preprocess_metadata_dataset(path_metadata)

    # Optional safety checks
    if validate_unique_sample_ids:
        if not meta.index.is_unique:
            raise ValueError("Metadata sample_id index contains duplicates.")
        if not taxa.index.is_unique:
            raise ValueError("Taxa sample_id index contains duplicates.")

    merged = meta.join(taxa, how=how)
    merged = merged.reset_index()  # bring sample_id back as a column
    return merged
