"""Document indexer: Google Drive + local files → Qdrant via llama-index."""
import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import io

GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
EMBED_DIM = 1536


class DocumentIndexer:
    def __init__(self, qdrant_url: str, collection: str):
        self._qdrant = QdrantClient(url=qdrant_url)
        self._collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if self._collection not in existing:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            print(f"[indexer] created collection '{self._collection}'")

    def _get_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(client=self._qdrant, collection_name=self._collection)

    def index_local_directory(self, directory: str | Path) -> int:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(directory)

        reader = SimpleDirectoryReader(str(directory), recursive=True)
        docs = reader.load_data()
        if not docs:
            print("[indexer] no documents found")
            return 0

        vector_store = self._get_vector_store()
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)
        print(f"[indexer] indexed {len(docs)} documents from {directory}")
        return len(docs)

    def index_gdrive(self, credentials_path: str, token_path: str = "token.json") -> int:
        creds = self._get_gdrive_creds(credentials_path, token_path)
        service = build("drive", "v3", credentials=creds)

        results = service.files().list(
            q="mimeType='application/vnd.google-apps.document' or mimeType='text/plain'",
            pageSize=100,
            fields="files(id, name, mimeType)",
        ).execute()
        files = results.get("files", [])

        tmp_dir = Path("/tmp/gdrive_docs")
        tmp_dir.mkdir(exist_ok=True)

        downloaded = 0
        for f in files:
            try:
                if f["mimeType"] == "application/vnd.google-apps.document":
                    content = service.files().export(
                        fileId=f["id"], mimeType="text/plain"
                    ).execute()
                    out_path = tmp_dir / f"{f['name']}.txt"
                    out_path.write_bytes(content)
                else:
                    request = service.files().get_media(fileId=f["id"])
                    buf = io.BytesIO()
                    downloader = MediaIoBaseDownload(buf, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    out_path = tmp_dir / f["name"]
                    out_path.write_bytes(buf.getvalue())
                downloaded += 1
            except Exception as e:
                print(f"[indexer] skipping {f['name']}: {e}")

        count = self.index_local_directory(tmp_dir)
        print(f"[indexer] downloaded {downloaded} from Google Drive, indexed {count}")
        return count

    @staticmethod
    def _get_gdrive_creds(credentials_path: str, token_path: str) -> Credentials:
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, GDRIVE_SCOPES)
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, GDRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
            Path(token_path).write_text(creds.to_json())
        return creds
