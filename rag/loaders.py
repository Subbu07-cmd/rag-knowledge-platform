"""Document loaders for local folders and S3 prefixes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".rst"}


@dataclass
class Document:
    text: str
    source: str


def load_local(folder: str | Path) -> list[Document]:
    root = Path(folder)
    return [
        Document(p.read_text(encoding="utf-8"), str(p.relative_to(root)))
        for p in sorted(root.rglob("*"))
        if p.suffix.lower() in TEXT_SUFFIXES
    ]


def load_s3(uri: str) -> list[Document]:
    import boto3

    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    docs: list[Document] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if Path(obj["Key"]).suffix.lower() in TEXT_SUFFIXES:
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode("utf-8")
                docs.append(Document(body, f"s3://{bucket}/{obj['Key']}"))
    return docs


def load(source: str) -> list[Document]:
    return load_s3(source) if source.startswith("s3://") else load_local(source)
