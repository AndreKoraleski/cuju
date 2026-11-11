# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the links of the README, the docs, AGENTS.md and CONTRIBUTING.md.

Run it as `python ci/check_links.py --offline` to check without the network, as CI does, that
every relative link names a file of the repository, that every link into this repository on
GitHub or Colab names a file that exists here, that every anchor into a Markdown file of this
repository names one of its headings, that every link into cuju-render names a path of
RENDER_PATHS below, and that no file mentions a thing of REMOVED. Run it by hand as
`python ci/check_links.py` to also request every external link once, HEAD first and GET when
HEAD fails, with a timeout, a pause between requests and a generic User-Agent, and to check each
path of RENDER_PATHS against cuju-render's repository on GitHub. The report names each link that
fails or moved permanently, with the files and lines that hold it.
"""

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'AndreKoraleski'
HERE = 'cuju'
RENDER = 'cuju-render'
BRANCH = 'main'

# Every path of cuju-render's repository that the files link to, as it follows
# https://github.com/AndreKoraleski/cuju-render/, with its anchor. The offline check holds each
# such link to this list, and the online check, run by hand, checks the list against
# cuju-render's repository on GitHub.
RENDER_PATHS = [
    '',
    'blob/main/docs/presentation.md',
    'blob/main/docs/reference.md',
    'blob/main/docs/rendering.md',
]

# The hosts whose addresses are no links to check, each with its reason.
SKIP = {
    '127.0.0.1': 'the address of a server on this computer',
    'localhost': 'the address of a server on this computer',
    'example.com': 'an address of the examples',
}

# The things that the files no longer mention, each with its reason.
REMOVED = {
    r'promo': 'the promotional montage and its script make_promo were removed',
    r'montage': 'the montage of the renderer left docs/assets',
    r'cuju-teaser': 'the teaser of the montage left docs/assets',
    r'mistermender|freesound\.org/s/512340': 'the music of the montage left with it',
}

# A generic client name, which names no person.
USER_AGENT = 'link-check/1.0 (Python urllib)'
TIMEOUT_SECONDS = 20
PAUSE_SECONDS = 0.5
REDIRECTS = 5

FENCE = re.compile(r'^\s{0,3}(```|~~~)')
CODE_SPAN = re.compile(r'(`+).+?\1')
INLINE = re.compile(r'\]\(\s*<?([^)\s>]+)>?')
REFERENCE = re.compile(r'^\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?')
ATTRIBUTE = re.compile(r'\b(?:href|src)\s*=\s*["\']([^"\']+)["\']')
BARE = re.compile(r'https?://[^\s<>"\'`)\]}|\\]+')
HEADING = re.compile(r'^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$')
NAMED = re.compile(r'<a\s+(?:id|name)="([^"]+)"')


@dataclass(frozen=True)
class Link:
    """A link as a file holds it."""

    file: Path
    line: int
    url: str

    def where(self, root):
        return f'{self.file.relative_to(root).as_posix()}:{self.line}'


def markdown_files(root):
    """Returns the README, AGENTS.md, CONTRIBUTING.md and the Markdown files of docs."""
    files = [root / name for name in ('README.md', 'AGENTS.md', 'CONTRIBUTING.md')]
    files += sorted((root / 'docs').rglob('*.md'))
    return [path for path in files if path.is_file()]


def clean(url):
    """Drops the punctuation of the sentence that ends a bare address."""
    url = url.rstrip('.,;:!?*_')
    while url.endswith(')') and url.count('(') < url.count(')'):
        url = url[:-1]
    return url


def markdown_links(path):
    links = []
    fenced = False
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        text = CODE_SPAN.sub('', line)
        found = INLINE.findall(text) + REFERENCE.findall(text) + ATTRIBUTE.findall(text)
        found += [clean(url) for url in BARE.findall(text)]
        for url in dict.fromkeys(found):
            links.append(Link(path, number, url))
    return links


def collect(root):
    return [link for path in markdown_files(root) for link in markdown_links(path)]


def slug(heading):
    """Returns the anchor that GitHub gives a heading of a Markdown file."""
    text = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', heading)
    text = re.sub(r'\[([^\]]*)\]\[[^\]]*\]', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text).replace('`', '').replace('*', '')
    text = re.sub(r'[^\w\- ]', '', text.strip().lower())
    return text.replace(' ', '-')


def anchors_of(text):
    """Returns the anchors of a Markdown text, its headings counted as GitHub counts them."""
    names, counts = set(), {}
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        heading = HEADING.match(line)
        if heading:
            name = slug(heading.group(2))
            seen = counts.get(name, 0)
            names.add(name if not seen else f'{name}-{seen}')
            counts[name] = seen + 1
        names.update(NAMED.findall(line))
    return names


class Anchors:
    """The anchors of the Markdown files of the repository, read once each."""

    def __init__(self):
        self.held = {}

    def of(self, path):
        if path not in self.held:
            self.held[path] = anchors_of(path.read_text(encoding='utf-8'))
        return self.held[path]


def repository_link(parsed):
    """Returns the repository and the rest of the path of a link into a repository of the owner,
    on GitHub, as a raw file or in Colab, a raw file written as a file on its branch."""
    parts = [urllib.parse.unquote(part) for part in parsed.path.split('/') if part]
    if parsed.hostname == 'colab.research.google.com' and parts[:1] == ['github']:
        parts = parts[1:]
    elif parsed.hostname not in ('github.com', 'raw.githubusercontent.com'):
        return None
    if len(parts) < 2 or parts[0] != OWNER:
        return None
    repository = parts[1].split('@')[0].removesuffix('.git')
    rest = parts[2:]
    if parsed.hostname == 'raw.githubusercontent.com':
        if rest[:2] == ['refs', 'heads']:
            rest = rest[2:]
        rest = ['blob', *rest]
    return repository, rest


def render_key(rest, fragment):
    return '/'.join(rest) + (f'#{fragment}' if fragment else '')


def problem_of(link, root, anchors, used):
    """Returns what is wrong with a link, or None, and notes each path of cuju-render it uses."""
    url = link.url
    if url.startswith(('mailto:', 'data:', 'tel:')):
        return None
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme in ('http', 'https'):
        if (parsed.hostname or '') in SKIP:
            return None
        target = repository_link(parsed)
        if target is None:
            return None
        repository, rest = target
        if repository == RENDER:
            key = render_key(rest, parsed.fragment)
            used.add(key)
            if key not in RENDER_PATHS:
                return f'links into cuju-render at {key!r}, which RENDER_PATHS does not list'
            return None
        if repository != HERE:
            return f'links into an unknown repository {repository!r}'
        if len(rest) < 2 or rest[0] not in ('blob', 'tree'):
            return None
        if rest[1] != BRANCH:
            return f'links to the branch {rest[1]!r}, where links into this repository name main'
        return local_problem(root.joinpath(*rest[2:]), parsed.fragment, root, anchors)
    if parsed.scheme or url.startswith('//'):
        return None
    target, _, fragment = url.partition('#')
    target = target.partition('?')[0]
    if not target:
        if fragment and fragment not in anchors.of(link.file):
            return f'names the anchor #{fragment}, which no heading of the file gives'
        return None
    if target.startswith('/'):
        path = root / urllib.parse.unquote(target.lstrip('/'))
    else:
        path = link.file.parent / urllib.parse.unquote(target)
    return local_problem(path, fragment, root, anchors)


def local_problem(path, fragment, root, anchors):
    if not path.exists():
        try:
            name = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            name = str(path)
        return f'names {name}, which the repository lacks'
    if fragment and path.suffix == '.md' and fragment not in anchors.of(path):
        return f'names the anchor #{fragment}, which no heading of {path.name} gives'
    return None


def offline_problems(root=ROOT):
    """Returns every problem that the check finds without the network, with its file and line."""
    root = Path(root)
    anchors = Anchors()
    used = set()
    problems = []
    for link in collect(root):
        problem = problem_of(link, root, anchors, used)
        if problem:
            problems.append(f'{link.where(root)}: {link.url}, {problem}')
    for key in RENDER_PATHS:
        if key not in used:
            problems.append(f'RENDER_PATHS lists {key!r}, which no link uses any more')
    patterns = [(re.compile(pattern, re.IGNORECASE), pattern) for pattern in REMOVED]
    for path in markdown_files(root):
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            for pattern, text in patterns:
                if pattern.search(line):
                    where = f'{path.relative_to(root).as_posix()}:{number}'
                    problems.append(f'{where}: mentions {text}, {REMOVED[text]}')
    return problems


class _Stay(urllib.request.HTTPRedirectHandler):
    """Hands every redirect back, so the check sees whether it is permanent."""

    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_Stay)


def fetch(url, method):
    """Returns the status, the Location or the error, and up to a megabyte of the body."""
    request = urllib.request.Request(
        url, method=method, headers={'User-Agent': USER_AGENT, 'Accept': '*/*'}
    )
    try:
        with _OPENER.open(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(1 << 20) if method == 'GET' else b''
            return response.status, None, body
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get('Location'), b''
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return None, str(getattr(error, 'reason', error)), b''


def check_online(url):
    """Returns what is wrong with an external address, or None, and the body of its answer."""
    current = url
    for _ in range(REDIRECTS + 1):
        status, detail, body = fetch(current, 'HEAD')
        if status is None or status >= 400:
            time.sleep(PAUSE_SECONDS)
            status, detail, body = fetch(current, 'GET')
        if status in (301, 308):
            return f'moved permanently to {urllib.parse.urljoin(current, detail or "")}', body
        if status in (302, 303, 307) and detail:
            current = urllib.parse.urljoin(current, detail)
            time.sleep(PAUSE_SECONDS)
            continue
        if status is None:
            return f'failed, {detail}', body
        if status >= 400:
            return f'answered {status}', body
        return None, body
    return f'redirected more than {REDIRECTS} times', b''


def render_problems():
    """Checks each path of RENDER_PATHS, its anchor included."""
    problems = []
    for key in RENDER_PATHS:
        rest, _, fragment = key.partition('#')
        url = f'https://github.com/{OWNER}/{RENDER}' + (f'/{rest}' if rest else '')
        problem, _ = check_online(url)
        time.sleep(PAUSE_SECONDS)
        parts = rest.split('/')
        if problem is None and fragment and parts[:2] == ['blob', BRANCH]:
            raw = f'https://raw.githubusercontent.com/{OWNER}/{RENDER}/{BRANCH}/'
            problem, body = check_online(raw + '/'.join(parts[2:]))
            time.sleep(PAUSE_SECONDS)
            if problem is None and fragment not in anchors_of(body.decode('utf-8', 'replace')):
                problem = f'has no anchor #{fragment}'
        if problem:
            problems.append(f'RENDER_PATHS {key!r}, {url}, {problem}')
    return problems


def online_problems(root=ROOT, report=print):
    root = Path(root)
    found = {}
    for link in collect(root):
        parsed = urllib.parse.urlsplit(link.url)
        if parsed.scheme in ('http', 'https') and (parsed.hostname or '') not in SKIP:
            found.setdefault(urllib.parse.urldefrag(link.url).url, []).append(link)
    report(f'Requesting {len(found)} external addresses')
    problems = []
    for url, links in sorted(found.items()):
        problem, _ = check_online(url)
        time.sleep(PAUSE_SECONDS)
        if problem:
            places = ', '.join(link.where(root) for link in links)
            problems.append(f'{url}, {problem}, in {places}')
    report(f'Checking the {len(RENDER_PATHS)} paths of cuju-render')
    return problems + render_problems()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--offline', action='store_true', help='make no request, as CI does')
    options = parser.parse_args(argv)
    problems = offline_problems()
    if not options.offline:
        problems += online_problems()
    for problem in problems:
        print(problem)
    print(f'{len(problems)} problems' if problems else 'Every link holds')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
