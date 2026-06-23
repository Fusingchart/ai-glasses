"""CLI helper to index a local directory or Google Drive into Qdrant."""
import argparse
import sys

from config import Config
from indexer import DocumentIndexer


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into Qdrant")
    sub = parser.add_subparsers(dest="cmd", required=True)

    local_p = sub.add_parser("local", help="Index a local directory")
    local_p.add_argument("directory", help="Path to document directory")

    gdrive_p = sub.add_parser("gdrive", help="Index Google Drive")
    gdrive_p.add_argument("credentials", help="Path to credentials.json")
    gdrive_p.add_argument("--token", default="token.json", help="Token cache path")

    args = parser.parse_args()
    cfg = Config()
    idx = DocumentIndexer(qdrant_url=cfg.qdrant_url, collection=cfg.qdrant_collection)

    if args.cmd == "local":
        count = idx.index_local_directory(args.directory)
        print(f"Indexed {count} documents.")
    elif args.cmd == "gdrive":
        count = idx.index_gdrive(args.credentials, args.token)
        print(f"Indexed {count} documents from Google Drive.")


if __name__ == "__main__":
    main()
