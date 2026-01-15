"""surya.datasets.helio_aws

This module provides an S3-focused dataset class that can be used as a drop-in
replacement for HelioNetCDFDataset when your index CSV contains S3 URIs.

Typical usage (child dataset):

    from infrastructure.datasets.helio_aws import HelioNetCDFDatasetAWS

    class MyDownstreamDataset(HelioNetCDFDatasetAWS):
        ...

The base class in helio.py supports both local paths and `s3://...` URIs. This wrapper class primarily:
  - sets AWS/S3-friendly defaults (e.g., enabling simplecache)
  - provides a single import target for “AWS mode” in downstream datasets

Notes on performance (AWS):
  - Run compute in the same AWS region as the bucket.
  - Prefer IAM roles over static credentials.
  - Keep `s3_use_simplecache=True` and place `s3_cache_dir` on fast ephemeral
    storage (instance store / NVMe) when possible.
"""

from __future__ import annotations

from typing import Optional

from workshop_infrastructure.datasets.helio import HelioNetCDFDataset
# from workshop_infrastructure.datasets.helio_boto import HelioNetCDFDataset


class HelioNetCDFDatasetAWS(HelioNetCDFDataset):
    """AWS/S3-oriented wrapper for :class:`~infrastructure.datasets.helio.HelioNetCDFDataset`.
    This class is intended to be imported and subclassed by downstream datasets
    (see template_dataset.py). It behaves identically to HelioNetCDFDataset, but
    enables S3 read-through caching by default.

    Parameters
    ----------
    All parameters are identical to HelioNetCDFDataset, with the addition that
    the following are expected to be used in AWS/S3 environments:

    s3_storage_options : dict | None, optional
        Passed to fsspec/s3fs. Use for settings like {'anon': True} for public
        buckets, or endpoint configuration for non-AWS S3 backends.
    s3_use_simplecache : bool, optional
        If True (default), use fsspec's simplecache to keep a local read-through
        cache of objects.
    s3_cache_dir : str, optional
        Directory used by simplecache. Default: /tmp/helio_s3_cache
    s3fs_kwargs : dict | None, optional
        Keyword args to s3fs.S3FileSystem (rarely needed when using IAM roles).
    """

    def __init__(
        self,
        *args,
        s3_storage_options: Optional[dict] = None,
        s3_use_simplecache: bool = False,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        s3fs_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            s3_storage_options=s3_storage_options,
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
            s3fs_kwargs=s3fs_kwargs,
            **kwargs,
        )
