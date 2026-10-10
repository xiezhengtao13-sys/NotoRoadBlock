"""Safely restore the original compact mask archive into the expected dataset layout."""
from pathlib import Path
import re, zipfile

ROOT = Path(__file__).resolve().parent.parent
DS = (ROOT / 'road_v2/dataset_noto2024_v0').resolve()
with zipfile.ZipFile(ROOT / 'data/noto_masks.zip') as archive:
    names = archive.namelist()
    assert all(re.fullmatch(r'masks/(road|slide_aerial|slide_lidar)/noto_\d{5}\.png', n) for n in names)
    assert archive.testzip() is None
    for name in names:
        destination = (DS / name).resolve()
        assert destination.is_relative_to(DS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != archive.read(name):
            raise RuntimeError(f'Refusing to overwrite a changed mask: {name}')
        destination.write_bytes(archive.read(name))
print(f'Restored {len(names)} original masks.')
