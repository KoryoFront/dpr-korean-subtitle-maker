#!/usr/bin/env python3
"""Rebuilds data/search.db, the one database the subtitle engine uses (it ships ready-made; you only need this to
refresh it). Downloads its sources into tools/sources/:
  - KK.db   : the DPRK Korean dictionary  (Samhung, decoded by digitalnk.com/dics)
  - KEEK.db : the DPRK Korean-English dictionary (same)
  - KP.zip  : DPRK place names from GeoNames (geonames.org, CC BY 4.0)
Tables: ke, kk (dictionaries), places (Korean name -> DPRK-style English: Sinhung County, Mt. Paektu),
loanwords (4,000+ DPRK loanwords with their origin and meaning: 세멘트 = cement, 뜨락또르 = tractor).
  python3 tools/build_dictionary.py
"""
import json, os, re, sqlite3, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'sources')
OUT = os.path.join(HERE, '..', 'data', 'search.db')


def fetch(url, name):
    os.makedirs(SRC, exist_ok=True)
    path = os.path.join(SRC, name)
    if not os.path.exists(path):
        print('downloading', url)
        urllib.request.urlretrieve(url, path + '.part'); os.replace(path + '.part', path)
    return path


kk_path = fetch('https://digitalnk.com/dics/KK.db', 'KK.db')
ke_path = fetch('https://digitalnk.com/dics/KEEK.db', 'KEEK.db')
zip_path = fetch('https://download.geonames.org/export/dump/KP.zip', 'KP.zip')
with zipfile.ZipFile(zip_path) as z:
    z.extract('KP.txt', SRC)

if os.path.exists(OUT):
    os.unlink(OUT)
db = sqlite3.connect(OUT)
db.execute('create table ke(word text, definition text)'); db.execute('create table kk(word text, definition text)')
for table, path in (('ke', ke_path), ('kk', kk_path)):
    src = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
    db.executemany('insert into %s values (?,?)' % table, src.execute('select word, definition from dictionary'))
db.execute('create index ke_w on ke(word)'); db.execute('create index kk_w on kk(word)')

# ---- loanwords: the Korean dictionary marks each with its origin, e.g. 세멘트 [cement《영》]
ke = dict(db.execute('select word, definition from ke'))
def gloss(w):
    r = ke.get(w)
    if not r: return None
    t = ' '.join(l for l in r.split('\n')[1:] if l and not re.fullmatch(r'\[[^\]]*\]', l))
    t = re.sub(r'《[^》]*》|\[[^\]]*\]|\([^)]*\)', '', t)
    t = re.sub(r'[①-⑳]', '', t)
    t = re.split(r'[;.]', t)[0].strip(' ,')
    t = re.sub(r'^(an?|the) ', '', t).strip()
    return t if re.search('[A-Za-z]', t) and not re.search('[가-힣]', t) else None
db.execute('create table loanwords(word text primary key, en text, origin text, lang text)')
rows = []
for word, d in db.execute('select word, definition from kk').fetchall():
    m = re.search(r'\[([^\]]*?)《([^》]+)》\]', d.split('\n', 1)[0])
    if m and re.fullmatch(r'[가-힣]{2,}', word):
        rows.append((word, gloss(word), m.group(1).lstrip('←').strip(), m.group(2)))
db.executemany('insert or replace into loanwords values (?,?,?,?)', rows)

# ---- place names (GeoNames), in DPRK house style
SUFFIX = ('구역', '로동자구', '시', '군', '도', '구', '리', '동', '읍', '산', '강', '호', '섬', '령', '봉')
RANK = {'ADM1': 1, 'PPLC': 1, 'ADM2': 2, 'PPLA': 2, 'PPLA2': 3, 'ADM3': 3, 'MT': 3, 'STM': 4, 'LK': 4, 'ISL': 4, 'PPL': 6}
MAJOR = {'ADM1', 'PPLC', 'ADM2', 'PPLA', 'PPLA2', 'ADM3', 'MT'}

def dprk_initial(h):
    """South Korean spelling drops an initial ㄹ/ㄴ (용남, 이천); the DPRK keeps it (룡남, 리천)."""
    s = ord(h[0]) - 0xAC00
    if not 0 <= s < 11172: return None
    cho, rest = divmod(s, 588)
    if cho == 11:                                   # ㅇ -> ㄹ (용 -> 룡, 이 -> 리)
        return chr(0xAC00 + 5 * 588 + rest) + h[1:]
    if cho == 2:                                    # ㄴ -> ㄹ (낙 -> 락)
        return chr(0xAC00 + 5 * 588 + rest) + h[1:]
    return None

def english(ascii_name):
    return re.sub(r"['`’]", '', ascii_name).strip()


def dprk_r(en):
    """The English of a name whose Korean keeps the DPRK initial ㄹ: Yongnam -> Ryongnam, Isan -> Risan, Nak -> Rak."""
    if en[:1] == 'Y': return 'R' + en[0].lower() + en[1:]
    if en[:1] in 'IN': return 'R' + (en[1:] if en[0] == 'N' else en[0].lower() + en[1:])
    return en


def house_style(en, code):
    """KCNA style: Sinhung County, Mt. Paektu, Hamhung (a city without -si)."""
    base = re.sub(r'-(gun|si|san|kuyok|guyok)$', '', en)
    if en.endswith('-gun'): return base + ' County'
    if en.endswith('-si'): return base
    if en.endswith('-san') and code in ('MT', 'HLL', 'PK'): return 'Mt. ' + base
    if en.endswith(('-kuyok', '-guyok')): return base + ' District'
    return en

db.execute('drop table if exists places')
db.execute('create table places(key text, english text, type text, rank int, major int, geonames text)')
rows = []
for line in open(os.path.join(SRC, 'KP.txt'), encoding='utf-8'):
    f = line.rstrip('\n').split('\t')
    code, en = f[7], english(f[2])
    hangul = {x.strip() for x in f[3].split(',') if re.fullmatch(r'[가-힣]{2,}', x.strip())}
    rank, major = RANK.get(code, 5), int(code in MAJOR)
    keys = set(hangul)
    for h in hangul:
        for suf in SUFFIX:
            if h.endswith(suf) and len(h) - len(suf) >= 2 and major:
                keys.add(h[:-len(suf)])
    for k in list(keys):
        v = dprk_initial(k) if en[:1] in 'RL' else None
        if v: keys.add(v)
    for k in keys:
        e = en
        if (ord(k[0]) - 0xAC00) // 588 == 5 and en[:1] in 'YIN':      # ㄹ-initial (DPRK) key: Ryongnam, not Yongnam
            e = dprk_r(en)
        rows.append((k, house_style(e, code), code, rank, major, en))
db.executemany('insert into places values (?,?,?,?,?,?)', rows)
db.execute('create index places_k on places(key)')
db.commit()
db.commit()
for t in ('ke', 'kk', 'places', 'loanwords'):
    print(t, db.execute('select count(*) from %s' % t).fetchone()[0])
print('written', os.path.abspath(OUT))
