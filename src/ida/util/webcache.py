import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve


@dataclass
class WebCache:
    cache_dir: Union[str, Path] = Path('~/.cache/ida/').expanduser()  # where to cache downloaded files

    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir).expanduser()

    def get_absolute_path(self, cached_file_name: str):
        path = self.cache_dir / cached_file_name
        assert path.exists()
        return path

    def download(self, file_name: str, download_url: str, download_name: Optional[str] = None):
        if download_name is None:
            download_name = file_name
        target_url = download_url + file_name
        logging.info('Downloading {}...'.format(target_url))
        urlretrieve(target_url, str(self.cache_dir / download_name))

    def cache(self, file_name: str, download_url: str, download_name: Optional[str] = None,
              is_archive=False) -> bool:
        """
        If not yet cached, download `file_name` from `download_url`.
        If `is_archive` is `True`, attempt to extract contents with tar.
        Returns `True` iff there was already a file called `file_name`
        in the cache.
        """
        if download_name is None:
            download_name = file_name

        self.cache_dir.mkdir(exist_ok=True)
        file_path = self.cache_dir / download_name
        if not file_path.exists():
            self.download(file_name, download_url, download_name)
            if is_archive:
                with tarfile.open(file_path) as archive_file:
                    archive_file.extractall(self.cache_dir)
                    file_path.unlink()
                    file_path.touch()  # leave proof that archive was extracted
            return False
        return True

    def open(self, file_name: str, download_url: Optional[str] = None, download_name: Optional[str] = None,
             *args, **kwargs):
        if download_name is None:
            download_name = file_name

        if download_url is not None:
            self.cache(file_name, download_url, download_name)
        return (self.cache_dir / download_name).open(*args, **kwargs)
