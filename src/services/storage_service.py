"""GCS Storage Service — temporary file hosting for audio and images.

Uploads files to Google Cloud Storage and generates signed URLs for
LINE to fetch (audio messages, generated images). Objects expire through
the bucket's lifecycle rule (Delete after 3 days, enforced by
scripts/deploy_cloud_run.sh); the app only deletes an object itself when
delivering it failed.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from uuid import uuid4

from google.auth import credentials as google_auth_credentials
from google.auth import iam
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account as google_service_account

from src.config import get_settings
from src.utils.gcp_metadata import get_service_account_email_sync
from src.utils.logger import logger


class StorageError(Exception):
    """Raised when storage operations fail."""


@dataclass
class UploadedMedia:
    """Represents a temporary GCS object exposed to LINE."""

    public_url: str
    blob_name: str | None = None
    size_bytes: int = 0
    duration_ms: int | None = None


class StorageService:
    """Upload temp files to GCS and return signed URLs."""

    def __init__(self) -> None:
        settings = get_settings()
        self._bucket_name = settings.gcs_bucket_name
        self._expiry_hours = settings.gcs_signed_url_expiry_hours
        self._current_month = datetime.now(UTC).strftime("%Y-%m")
        self._month_upload_count = 0
        self._month_upload_bytes = 0
        self._month_delete_count = 0
        self._credentials = None
        self._service_account_email = ""
        self._bucket = None
        self._signing_credentials: google_auth_credentials.Signing | None = None

        if self._bucket_name:
            try:
                from google.cloud import storage
                client = storage.Client()
                self._bucket = client.bucket(self._bucket_name)
                self._credentials = getattr(client, "_credentials", None)
                self._service_account_email = (
                    getattr(self._credentials, "service_account_email", "") or ""
                )
                # On Cloud Run, compute credentials report "default" as the email.
                # Resolve the real email from the metadata server for signBlob to work.
                if not self._service_account_email or self._service_account_email == "default":
                    self._service_account_email = get_service_account_email_sync()
                signed_url_mode = "private-key credentials"
                if (
                    self._credentials is not None
                    and not isinstance(self._credentials, google_auth_credentials.Signing)
                ):
                    if self._service_account_email:
                        signed_url_mode = f"IAM Signer ({self._service_account_email})"
                    else:
                        signed_url_mode = "unavailable"
                logger.info(
                    "StorageService initialized, "
                    f"bucket={self._bucket_name}, "
                    f"signed_url_mode={signed_url_mode}"
                )
            except Exception as e:
                logger.warning(f"StorageService GCS init failed: {e}")
        else:
            logger.warning("StorageService: GCS_BUCKET_NAME not configured")

    @property
    def is_configured(self) -> bool:
        return self._bucket is not None

    def _check_and_reset_month(self) -> None:
        current_month = datetime.now(UTC).strftime("%Y-%m")
        if current_month != self._current_month:
            logger.info(
                f"GCS usage stats reset: {self._current_month} -> {current_month}, "
                f"uploads={self._month_upload_count}, "
                f"upload_bytes={self._month_upload_bytes}, "
                f"deletes={self._month_delete_count}"
            )
            self._current_month = current_month
            self._month_upload_count = 0
            self._month_upload_bytes = 0
            self._month_delete_count = 0

    def record_upload(self, size_bytes: int) -> None:
        self._check_and_reset_month()
        self._month_upload_count += 1
        self._month_upload_bytes += size_bytes

    def record_delete(self) -> None:
        self._check_and_reset_month()
        self._month_delete_count += 1

    def get_usage_stats(self) -> dict:
        self._check_and_reset_month()
        return {
            "configured": self.is_configured,
            "bucket_name": self._bucket_name or None,
            "month": self._current_month,
            "uploads": self._month_upload_count,
            "upload_bytes": self._month_upload_bytes,
            "deleted_objects": self._month_delete_count,
            "signed_url_expiry_hours": self._expiry_hours,
            "media_expiry": "bucket lifecycle",
            "scope": "per-instance",
        }

    def _signed_url_kwargs(self) -> dict:
        kwargs = {
            "expiration": timedelta(hours=self._expiry_hours),
            "version": "v4",
        }
        credentials = self._signing_credentials_for_urls()
        if credentials is None:
            return kwargs

        kwargs["credentials"] = credentials
        return kwargs

    def _signing_credentials_for_urls(self) -> google_auth_credentials.Signing | None:
        credentials = self._credentials
        if credentials is None:
            return None
        if isinstance(credentials, google_auth_credentials.Signing):
            return credentials

        if not self._service_account_email:
            raise StorageError(
                "Storage credentials cannot sign URLs because no service account email "
                "is available."
            )

        if self._signing_credentials is None:
            # storage.Client credentials have devstorage-only scopes which
            # cannot call iamcredentials.googleapis.com/signBlob.  We need
            # cloud-platform scoped credentials for the IAM signer.
            import google.auth
            iam_credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            signer = iam.Signer(
                GoogleAuthRequest(),
                iam_credentials,
                self._service_account_email,
            )
            self._signing_credentials = google_service_account.Credentials(
                signer=signer,
                service_account_email=self._service_account_email,
                token_uri="https://oauth2.googleapis.com/token",
            )

        return self._signing_credentials

    def _generate_signed_url(self, blob) -> str:
        try:
            return blob.generate_signed_url(**self._signed_url_kwargs())
        except StorageError:
            raise
        except Exception as e:
            message = str(e)
            logger.error(
                f"Signed URL generation failed: {type(e).__name__}: {message}"
            )
            if any(
                token in message
                for token in (
                    "signBlob",
                    "signBytes",
                    "iamcredentials",
                    "private key to sign credentials",
                )
            ):
                raise StorageError(
                    "Failed to generate a signed GCS URL. Ensure "
                    "iamcredentials.googleapis.com is enabled and the Cloud Run runtime "
                    "service account has roles/iam.serviceAccountTokenCreator on itself. "
                    f"Original error: {type(e).__name__}: {message}"
                ) from e
            raise

    async def upload_base64_image(self, data_url: str) -> UploadedMedia | None:
        """Upload a base64 data URL image to GCS and return a signed URL.

        Args:
            data_url: base64 data URL like "data:image/png;base64,..."
                     or raw base64 string

        Returns:
            Signed URL or None on failure
        """
        if not self.is_configured:
            logger.warning("GCS not configured, cannot upload image")
            return None

        try:
            # Parse data URL
            if data_url.startswith("data:"):
                header, b64data = data_url.split(",", 1)
                content_type = header.split(":")[1].split(";")[0]
                ext = content_type.split("/")[1]
            else:
                b64data = data_url
                content_type = "image/png"
                ext = "png"

            image_bytes = base64.b64decode(b64data)
            return await self._upload_bytes(
                image_bytes, f"img/{uuid4()}.{ext}", content_type
            )
        except Exception as e:
            logger.error(f"Failed to upload base64 image: {e}")
            return None

    async def upload_file(self, file_path: str, content_type: str) -> UploadedMedia | None:
        """Upload a local file to GCS and return a signed URL.

        Args:
            file_path: Path to the file
            content_type: MIME type (e.g. "audio/mpeg")

        Returns:
            Signed URL or None on failure
        """
        if not self.is_configured:
            logger.warning("GCS not configured, cannot upload file")
            return None

        try:
            ext = os.path.splitext(file_path)[1] or ".bin"
            blob_name = f"audio/{uuid4()}{ext}"
            size_bytes = os.path.getsize(file_path)

            loop = asyncio.get_running_loop()

            # Run synchronous GCS upload in executor to avoid blocking event loop
            blob = self._bucket.blob(blob_name)
            await loop.run_in_executor(
                None,
                partial(blob.upload_from_filename, file_path, content_type=content_type),
            )

            url = await loop.run_in_executor(
                None,
                partial(self._generate_signed_url, blob),
            )

            self.record_upload(size_bytes)
            logger.info(f"Uploaded {blob_name}, URL expires in {self._expiry_hours}h")
            return UploadedMedia(public_url=url, blob_name=blob_name, size_bytes=size_bytes)
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return None

    async def _upload_bytes(
        self, data: bytes, blob_name: str, content_type: str
    ) -> UploadedMedia | None:
        """Upload raw bytes to GCS."""
        try:
            loop = asyncio.get_running_loop()

            blob = self._bucket.blob(blob_name)
            await loop.run_in_executor(
                None,
                partial(blob.upload_from_string, data, content_type=content_type),
            )

            url = await loop.run_in_executor(
                None,
                partial(self._generate_signed_url, blob),
            )

            self.record_upload(len(data))
            logger.info(f"Uploaded {blob_name}, URL expires in {self._expiry_hours}h")
            return UploadedMedia(public_url=url, blob_name=blob_name, size_bytes=len(data))
        except Exception as e:
            logger.error(f"Failed to upload bytes: {e}")
            return None

    async def delete_blob(self, blob_name: str) -> bool:
        """Delete a temporary object from GCS."""
        if not self.is_configured:
            return False

        try:
            loop = asyncio.get_running_loop()
            blob = self._bucket.blob(blob_name)
            await loop.run_in_executor(None, blob.delete)
            self.record_delete()
            logger.info(f"Deleted temporary object: {blob_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete temporary object {blob_name}: {e}")
            return False


# Global singleton
_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
