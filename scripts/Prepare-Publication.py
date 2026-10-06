"""Prepare a new local source snapshot for review; never creates a remote or pushes.

Raw runs and operational logs stay in the development checkout. Markdown links
to excluded local evidence become explicit local-only references. The source
checkout and its Git index/history are left untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {'adapters', 'agents', 'sim', 'experiments', 'scripts', 'tests', 'web', 'docs'}
EXTENSIONS = {'.py', '.ps1', '.cmd', '.js', '.css', '.html', '.md', '.svg', '.json', '.jpg'}
ROOT_FILES = {'.gitignore', 'LICENSE', 'NOTICE', 'README.md', 'CHANGELOG.md'}
EXCLUDED = {'AGENTS.md', 'docs/POPULATION-WATCH-LOG.md'}


def prepare(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'local') or output.exists():
        raise ValueError('Choose a new directory inside this checkout\'s ignored local/ directory')
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    names = sorted(set(tracked.decode('utf-8').split('\0')) - {''})
    selected = []
    for name in names:
        path = Path(name)
        if name in EXCLUDED:
            continue
        allowed = (name in ROOT_FILES or
                   len(path.parts) == 1 and path.suffix in {'.py', '.cmd'} or
                   path.parts[0] in DIRECTORIES and path.suffix in EXTENSIONS)
        source = ROOT / path
        if allowed and source.is_file():
            if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
                raise ValueError(f'Unsupported linked file: {name}')
            selected.append(name)
    included = {(ROOT / name).resolve() for name in selected}
    source_root = output / 'source'
    source_root.mkdir(parents=True)
    changed = {}
    files = []
    for name in selected:
        source = ROOT / name
        body = source.read_bytes()
        substitutions = []
        if source.suffix == '.md':
            try:
                text = body.decode('utf-8-sig')
            except UnicodeDecodeError:
                text = body.decode('cp1252')
                substitutions.append('Windows-1252 text converted to UTF-8 in public copy')

            def link(match):
                label, raw = match.group(1), match.group(2).strip().strip('<>')
                target = raw.split('#', 1)[0]
                if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', target):
                    return match.group(0)
                if (source.parent / target).resolve() not in included:
                    substitutions.append(raw)
                    return f'{label} (local evidence; not bundled)'
                return match.group(0)

            text = re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)', link, text)
            text, count = re.subn(r'''[A-Za-z]:[/\\]Users[/\\][^/\\\s`'"<>]+''', '<user-home>', text)
            if count:
                substitutions.append(f'{count} machine-specific home path(s) replaced')
            body = text.encode('utf-8')
        if substitutions:
            changed[name] = substitutions
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        files.append({'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                      'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    review = {'title': 'Blissful Ignorance', 'proposed_repository': 'Maverick0351a/blissful-ignorance',
              'visibility': 'public (not published)', 'license': 'Apache-2.0',
              'files': files, 'file_count': len(files), 'bytes': sum(row['bytes'] for row in files),
              'documentation_adjustments': changed,
              'excluded': sorted(EXCLUDED) + ['runs/', 'local/', 'BUILD-PLAN.md', 'all ignored files',
                         'existing Git history', 'model weights', 'live saves and private journals'],
              'remote_created': False, 'pushed': False}
    (output / 'review.json').write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'snapshot': str(output.relative_to(ROOT)), 'files': len(files),
                      'bytes': review['bytes'], 'documents_adjusted': len(changed), 'published': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    prepare(parser.parse_args().output)
