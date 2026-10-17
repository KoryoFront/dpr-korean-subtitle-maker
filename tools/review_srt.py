#!/usr/bin/env python3
"""Checks a subtitle file against the house rules and lists everything that breaks them.
  python3 review_srt.py english.srt"""
import os, re, sys, collections

for d in (os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.exists(os.path.join(d, 'subtitles.py')):
        sys.path.insert(0, d)
        break
from subtitles import in_name, LEADERS               # the same name rule the subtitles are made with

SOUTH = r"\b(Baekdu|Joseon|Pyeongyang|Hamheung|Kaeseong|Gaeseong|Shinuiju|Sinuiju-si|Cheongjin|Jeongju|Kim Jong-un|Kim Il-sung|Kim Jong-il|Kim Yo-jong|Ri Il-hwan|Chosun|Daedong|Busan)\b"
FILLER = r"(?<![\w'])(uh+|um+|er+m?|ern|ah+|eh+|hmm+)(?![\w'])"


def t(x):
    h, m, s = x.replace(',', '.').split(':'); return int(h) * 3600 + int(m) * 60 + float(s)


def main(path):
    words = set(w.strip().lower() for w in open('/usr/share/dict/words')) if __import__('os').path.exists('/usr/share/dict/words') else set()
    cues = []
    for block in open(path, encoding='utf-8').read().strip().split('\n\n'):
        lines = block.split('\n')
        if len(lines) >= 3 and '-->' in lines[1]:
            a, b = lines[1].split(' --> ')
            cues.append((t(a), t(b), lines[2:]))
    problems = collections.defaultdict(list)
    prev = None
    for i, (a, b, rows) in enumerate(cues, 1):
        plain = [re.sub(r'</?b>', '', r) for r in rows]
        text = ' '.join(plain)
        song = text.startswith('♪')
        body = [r.strip('♪ ').strip() for r in plain]
        if len(rows) > 2: problems['more than 2 lines'].append((i, text))
        if any(len(r) > 37 for r in body): problems['line over 37 characters'].append((i, text))
        if len(rows) == 2 and any(len(r.split()) == 1 for r in body): problems['one-word line'].append((i, text))
        if b - a < 0.82: problems['on screen under 0.83 s'].append((i, text))
        if b - a > 7.05: problems['on screen over 7 s'].append((i, text))
        if prev and a < prev[1] - 0.01: problems['overlaps the one before'].append((i, text))
        if re.search(SOUTH, text): problems['South Korean spelling'].append((i, text))
        if re.search(r'North Korea', text): problems['"North Korea"'].append((i, text))
        if re.search(FILLER, text, re.I): problems['filler word'].append((i, text))
        if prev and text == prev[2]: problems['same as the one before'].append((i, text))
        if len(rows) == 2:
            w = rows[0].split() + rows[1].split()
            w = [re.sub(r'</?b>', '', x) for x in w]
            if in_name(w, len(rows[0].split())): problems['name split over two lines'].append((i, ' / '.join(plain)))
        if prev and not song:
            w = prev[2].split() + text.split()
            if in_name(w, len(prev[2].split())): problems['name split over two subtitles'].append((i, prev[2][-30:] + ' | ' + text[:30]))
        for m in LEADERS.finditer(text):
            if '<b>%s</b>' % m.group(0) not in ' '.join(rows):
                problems["leader's name not bold"].append((i, text))
        if re.search(r'[;:-]$', body[-1]): problems['line ends with ; : or -'].append((i, text))
        if words:
            odd = [w for w in re.findall(r"\b[a-z][a-z]+\b", text) if not any(x in words for x in (w, w[:-1], w[:-2], w[:-3]))]
            if odd: problems['word not in the English dictionary'].append((i, ' '.join(odd) + '  |  ' + text))
        prev = (a, b, text)
    songs = sum(1 for c in cues if ' '.join(c[2]).startswith('♪'))
    print('%d subtitles, %d of them song lyrics, %.1f h covered' % (len(cues), songs, sum(b - a for a, b, _ in cues) / 3600))
    for k, v in sorted(problems.items(), key=lambda kv: -len(kv[1])):
        print('\n== %s: %d' % (k, len(v)))
        for i, x in v[:12]:
            print('  #%d  %s' % (i, x[:150]))
    return problems


if __name__ == '__main__':
    main(sys.argv[1])
