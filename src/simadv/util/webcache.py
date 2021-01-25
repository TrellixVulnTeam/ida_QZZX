import logging
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


class WebCache:

    def __init__(self, cache_dir):
        """
        :param cache_dir: where to cache downloaded files
        """
        self.cache_dir = Path(cache_dir).expanduser()

    def get_absolute_path(self, cached_file_name: str):
        path = self.cache_dir / cached_file_name
        assert path.exists()
        return path

    def download(self, file_name: str, download_url: str):
        target_url = download_url + file_name
        logging.info('Downloading {}...'.format(target_url))
        urlretrieve(target_url, str(self.cache_dir / file_name))

    def cache(self, file_name: str, download_url: str,
              is_archive=False):
        """
        If not yet cached, download `file_name` from `download_url`.
        If `is_archive` is `True`, attempt to extract contents with tar.
        Returns `True` iff there was already a file called `file_name`
        in the cache.
        """
        self.cache_dir.mkdir(exist_ok=True)
        file_path = self.cache_dir / file_name
        if not file_path.exists():
            self.download(file_name, download_url)
            if is_archive:
                with tarfile.open(file_path) as archive_file:
                    archive_file.extractall(self.cache_dir)
                    file_path.unlink()
                    file_path.touch()  # leave proof that archive was extracted
            return False
        return True

    def open(self, file_name: str, download_url: str=None, *args, **kwargs):
        if download_url is not None:
            self.cache(file_name, download_url)
        return (self.cache_dir / file_name).open(*args, **kwargs)
