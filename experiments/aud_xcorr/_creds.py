"""Resolve AWS creds synchronously via boto3 (credential_process) and export
them as env vars so aiobotocore/s3fs doesn't need to spawn a subprocess
(which fails on Windows' selector event loop). Import before any S3 access."""
import os
import pathlib
import tempfile

import boto3

_c = boto3.Session().get_credentials().get_frozen_credentials()
os.environ["AWS_ACCESS_KEY_ID"] = _c.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = _c.secret_key
if _c.token:
    os.environ["AWS_SESSION_TOKEN"] = _c.token
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
_cfg = pathlib.Path(tempfile.gettempdir()) / "aws_config_no_credproc"
_cfg.write_text("[default]\nregion = us-west-2\n")
os.environ["AWS_CONFIG_FILE"] = str(_cfg)  # stop credential_process being used
