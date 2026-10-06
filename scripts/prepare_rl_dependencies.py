"""Fetch only the four explicitly approved wheels; inspect without execution.

Downloading requires the user's separate approval. This script does not install,
import or extract their code. Installation and focused source review are separate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--download-approved', action='store_true')
    args = parser.parse_args()
    entries = json.loads(Path(__file__).with_name('rl-dependencies.json').read_text())
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = []
    for item in entries:
        path = directory/item['filename']
        if not path.exists():
            if not args.download_approved:
                raise RuntimeError('Missing wheel; download approval is required')
            with urllib.request.urlopen(item['url'], timeout=45) as response:
                data = response.read(item['size']+1)
            if len(data) != item['size'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise RuntimeError('Download size or digest mismatch')
            with path.open('xb') as handle:
                handle.write(data)
        data = path.read_bytes()
        assert len(data) == item['size'] and hashlib.sha256(data).hexdigest() == item['sha256']
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            seen = set()
            python_files = 0
            for member in archive.infolist():
                name = member.filename
                parts = PurePosixPath(name)
                assert not parts.is_absolute() and not {'..', '.'} & set(parts.parts)
                assert ':' not in name and '\\' not in name and name.casefold() not in seen
                assert not stat.S_ISLNK(member.external_attr >> 16) and not member.flag_bits & 1
                assert not name.lower().endswith(('.pth', '.exe', '.dll', '.pyd', '.so'))
                assert all(not p.endswith((' ', '.')) for p in parts.parts)
                seen.add(name.casefold())
                if name.endswith('.py'):
                    ast.parse(archive.read(member), filename=name)
                    python_files += 1
            wheel = next(n for n in archive.namelist() if n.endswith('.dist-info/WHEEL'))
            assert 'Root-Is-Purelib: true' in archive.read(wheel).decode()
            report.append(dict(item, entries=len(seen), python_files=python_files,
                crc_valid=True, paths_valid=True, pure_python=True,
                unpacked_bytes=sum(i.file_size for i in archive.infolist())))
    (directory/'VERIFIED.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
