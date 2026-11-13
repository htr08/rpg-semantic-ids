"""Kiểm tra tính toàn vẹn và việc giữ file tải lỗi ngoài đường dẫn dữ liệu chính."""

import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_data import download_file, inspect_file


class DownloadDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = gzip.compress(b'{"asin":"A"}\n{"asin":"B"}\n', mtime=0)
        self.source = self.root / "source.gz"
        self.source.write_bytes(self.payload)
        self.spec = {
            "filename": "reviews.gz",
            "url": self.source.as_uri(),
            "size_bytes": len(self.payload),
            "records": 2,
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }

    def test_download_and_existing_file_need_no_network(self):
        result = download_file(self.spec, self.root / "raw", retries=1)
        self.assertEqual(result["records"], 2)
        self.assertFalse((self.root / "raw" / "reviews.gz.part").exists())
        with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network")):
            self.assertEqual(download_file(self.spec, self.root / "raw"), result)

    def test_bad_checksum_never_publishes_download(self):
        spec = {**self.spec, "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            download_file(spec, self.root / "raw", retries=1)
        self.assertFalse((self.root / "raw" / "reviews.gz").exists())
        self.assertFalse((self.root / "raw" / "reviews.gz.part").exists())

    def test_truncated_download_rejected(self):
        self.source.write_bytes(self.payload[:-5])
        with self.assertRaisesRegex(ValueError, "size"):
            download_file(self.spec, self.root / "raw", retries=1)

    def test_crc_corruption_detected_without_reference_hash(self):
        corrupted = bytearray(self.payload)
        corrupted[-8] ^= 1
        self.source.write_bytes(corrupted)
        with self.assertRaises(gzip.BadGzipFile):
            inspect_file(self.source, {**self.spec, "sha256": None})

    def test_wrong_record_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "records"):
            inspect_file(self.source, {**self.spec, "records": 3})


if __name__ == "__main__":
    unittest.main()
