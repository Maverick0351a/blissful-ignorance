"""Build a static, recorded Hugging Face showcase from a reviewed Git commit.

Reads only an explicit list of already-published assets. Does not read live
saves, download anything, run models, create a Space or upload files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'https://github.com/Maverick0351a/blissful-ignorance'
ASSETS = {
    name: 'docs/progress/' + name for name in (
        'progress.css', 'progress.js', 'results.js', 'results.json',
        'replay.js', 'world-preview.jpg',
    )
}
ASSETS.update({'icon.svg': 'web/icon.svg', 'LICENSE': 'LICENSE', 'NOTICE': 'NOTICE'})


def build(checkout, commit, output):
    checkout, output = checkout.resolve(), output.resolve()
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Use the complete reviewed Git commit ID.')
    if not output.is_relative_to(ROOT / 'local') or output.exists():
        raise ValueError('Choose a new output directory inside ignored local/.')

    def read(name):
        return subprocess.check_output(['git', 'show', f'{commit}:{name}'],
                                       cwd=checkout, timeout=15)

    files = {name: read(original) for name, original in ASSETS.items()}
    page = read('docs/progress/index.html').decode('utf-8')
    page = page.replace('<title>Blissful Ignorance · Progress</title>',
                        '<title>Blissful Ignorance · Recorded showcase</title>')
    page = page.replace('Development journal <b>01</b>', 'Recorded showcase <b>01</b>')
    page = page.replace('<a href="../../" id="world-link" hidden>← The world</a>',
                        f'<a href="{REPOSITORY}#play-locally" id="world-link" '
                        'target="_blank" rel="noopener noreferrer">Play locally ↗</a>')

    def link(match):
        target = match.group(1)
        if target == '../../web/icon.svg':
            return 'href="icon.svg"'
        if target.endswith('.md'):
            name = posixpath.normpath(posixpath.join('docs/progress', target))
            read(name)  # Every linked document must exist in the source commit.
            url = REPOSITORY if name == 'README.md' else f'{REPOSITORY}/blob/{commit}/{name}'
            return f'href="{url}" target="_blank" rel="noopener noreferrer"'
        return match.group(0)

    page = re.sub(r'href="([^"]+)"', link, page)
    page = page.replace('src="../../web/icon.svg"', 'src="icon.svg"')
    intro = '''
    <section class="hosted-intro wrap" aria-label="About this recorded showcase">
      <div class="hosted-heading"><div><p class="eyebrow">ARTIFICIAL LIFE · RESEARCH PROTOTYPE</p>
      <h1>Blissful Ignorance</h1></div><span class="pill">Recorded showcase</span></div>
      <p>Meet a small world of independent learning agents. Explore the measured
      results and replay a completed experiment below. This page does not run
      or train a live population.</p>
      <figure><img src="world-preview.jpg" width="1265" height="712"
      alt="The playable local valley with eight named characters, their map and individual senses.">
      <figcaption>Screenshot of a separate playable test world. The game runs
      locally; the interactive playback below shows a recorded feeding experiment.</figcaption></figure>
      <div class="hero-actions"><a class="button" href="#replay">Watch the recorded trial ↗</a>
      <a class="text-link" href="REPOSITORY#play-locally" target="_blank"
      rel="noopener noreferrer">Get the playable world ↗</a></div>
    </section>'''.replace('REPOSITORY', REPOSITORY)
    page = page.replace('<main id="main">', '<main id="main">' + intro)
    page = page.replace('<h1>Learning to feed<br>themselves.</h1>',
                        '<h2 class="recorded-title">Learning to feed<br>themselves.</h2>')
    page = page.replace('The live population uses eight independent recurrent PPO models and a distinct local NPU Laya model with frozen weights and private context.',
                        'At the source snapshot date, the private local population had eight independent recurrent PPO models and a distinct local NPU Laya model with frozen weights and private context.')
    page = page.replace('No trial weights have been promoted.',
                        'The recorded trial weights had not been promoted into that local population.')
    files['index.html'] = page.encode('utf-8')
    script = files['progress.js'].decode('utf-8')
    original = '$("world-link").hidden = !(/^https?:$/.test(location.protocol) && new URLSearchParams(location.search).get("from") === "world");'
    if original not in script:
        raise ValueError('The source navigation changed; review the static-page adaptation.')
    script = script.replace('// Only show the return link when opened from the running game\'s own navigation.',
                            '// The hosted recorded showcase links to the playable source.')
    files['progress.js'] = script.replace(original, '$("world-link").hidden = false;').encode('utf-8')
    files['progress.css'] += b'''
html{scroll-padding-top:80px}
.masthead{padding-top:68px;padding-bottom:18px;min-height:0;align-items:flex-end}
.skip:focus{top:68px}
.hosted-intro{padding-top:46px;padding-bottom:24px}
.hosted-heading{display:flex;align-items:center;justify-content:space-between;gap:24px}
.hosted-intro h1{font-size:clamp(36px,5vw,64px);line-height:1.08;margin:8px 0 14px}
.hosted-intro>p{max-width:760px;line-height:1.7}
.hosted-intro figure{margin:24px 0}.hosted-intro img{display:block;width:100%;height:auto;border-radius:12px}
.hosted-intro figcaption{font-size:13px;line-height:1.6;margin-top:10px;color:var(--muted)}
.hosted-intro+.hero{padding-top:42px}.recorded-title{font-size:clamp(32px,4.6vw,56px);line-height:1.08;margin:12px 0 22px}
@media(max-width:640px){.hosted-intro{padding-top:28px}.hosted-heading{align-items:flex-start;flex-direction:column;gap:8px}}
'''
    files['README.md'] = f'''---
title: Blissful Ignorance
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
fullWidth: true
header: mini
license: apache-2.0
short_description: Recorded learning experiments in a small world
tags:
  - artificial-life
  - reinforcement-learning
  - recorded-demo
---

# Blissful Ignorance: recorded showcase

A free interactive presentation of recorded experiments from a playable
artificial-life world. This Space has charts, per-learner filters and playback
of a completed feeding experiment. It does not run or train a live population.

The map image is a screenshot of a separate local test world. The replay uses
recorded experimental states and a simplified diagram, not the full game renderer.

[Get the playable source]({REPOSITORY}#play-locally) ·
[Experiment and limitations]({REPOSITORY}/blob/{commit}/docs/PRACTICE-AMOUNT.md) ·
[Milestones]({REPOSITORY}/blob/{commit}/docs/MILESTONES.md)

Source snapshot: [`{commit[:7]}`]({REPOSITORY}/commit/{commit}), October 6, 2026.
Six retained category-PPO brains improved prompt nearby-food acquisition from
75/96 to 95/96 evaluation lives after additional practice. The starting policy
was already trained. These results do not establish navigation, sustained
survival, meaningful communication or general intelligence; GT-01 remains open.

Only already-published assets are included. No private saves, raw training
traces, model weights, local credentials or operational logs are uploaded.
The source includes Apache-2.0 license and notice files. This static app needs
no package installation, external inference service, GPU or runtime secret.

The 217-test figure on the recorded result refers to that experiment's original
suite. The later GitHub source snapshot passed 231 tests after preservation fixes.
'''.encode('utf-8')
    destination = output / 'source'
    destination.mkdir(parents=True)
    for name, data in files.items():
        (destination / name).write_bytes(data)
    review = {'repository': REPOSITORY, 'source_commit': commit, 'kind': 'static-recorded-showcase',
              'sdk': 'static', 'live_models': False, 'new_downloads': False,
              'files': [{'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                        for name, data in sorted(files.items())]}
    review.update(file_count=len(files), bytes=sum(len(data) for data in files.values()))
    (output / 'review.json').write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'files': len(files), 'bytes': review['bytes'], 'published': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Reviewed public Git checkout')
    parser.add_argument('--commit', required=True, help='Published 40-character Git commit')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.source, args.commit, args.output)
