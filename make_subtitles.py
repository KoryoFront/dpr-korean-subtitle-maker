#!/usr/bin/env python3
"""
KCTV English subtitles for a recording on your own computer.

Asks for the path of a recording (.ts, .mkv, .mp4 ...) and a Google AI Studio (Gemini) API key, then:
  1. straightens the timestamps (lossless stream copy, no re-encode)
  2. optionally trims to the broadcast: drops the off-air test signals (1 kHz tone, two-tone signal) before and after
  3. sends the sound, in 36-second chunks and in parallel, to Gemini, which transcribes the Korean and translates it
     into English in DPRK / KCNA house style (Kim Jong Un, Ri Il Hwan, Mt. Paektu, never "North Korea")
  4. writes <name>_English.srt and <name>_English.mkv (picture and sound copied untouched + an English subtitle track)
     and checks that every picture and sound packet made it into the MKV

Needs: Python 3.8+, numpy (pip install numpy) and ffmpeg/ffprobe on the PATH (https://ffmpeg.org/download.html).
Progress is saved in a work folder next to the recording, so if it stops you can run it again and it carries on.
Disk space: about twice the size of the recording while it runs (the work folder is removed at the end).
"""
import base64, getpass, json, os, re, shutil, subprocess, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except ImportError:
    sys.exit('numpy is missing: run  pip install numpy  and start again.')

MODEL = 'gemini-flash-lite-latest'     # fast and cheap; quality was as good as Whisper large-v3 in tests
VERTEX_MODELS = {'gemini-flash-lite-latest': 'gemini-3.5-flash-lite'}   # Vertex has no "-latest" alias


def gemini_request(key, model, body):
    """A Vertex AI key (starts "AQ.", billed to a Google Cloud project, so Cloud credits apply) goes to Vertex AI, an
    AI Studio key to the Gemini API: same model, same request (Vertex wants each message's role)."""
    if str(key).startswith('AQ.'):
        b = json.loads(body)
        for c in b.get('contents', []):
            c.setdefault('role', 'user')
        url = 'https://aiplatform.googleapis.com/v1/publishers/google/models/%s:generateContent' % VERTEX_MODELS.get(model, model)
        body = json.dumps(b).encode()
    else:
        url = 'https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent' % model
    return urllib.request.Request(url, data=body, headers={'x-goog-api-key': key, 'Content-Type': 'application/json'})
CHUNK_S = 36                           # seconds of sound per request: as live, short for precise timing
PARALLEL = 16                          # requests in flight (a paid AI Studio key handles this; free tier will wait and retry)
SR = 16000

# ------------------------------------------------------------------ the English house style
DPRK_STYLE = (
    "ROMANIZATION of Korean proper names not given elsewhere (places, factories, mines, works): the DPRK system, McCune-Reischauer without breves: ㅓ and ㅗ = o, ㅡ and ㅜ = u, ㅈ = j, ㅊ = ch, ㅌ = th, ㅍ = ph, ㅋ = kh, initial ㄱ ㄷ ㅂ = k t p (제남 = Jenam, 백화원 = Paekhwawon, 덕천 = Tokchon, 화성의숙 = Hwasong Uisuk), names written as one word (Ryugyong, Chollima). Fixed titles: 《ㅌ.ㄷ》 = the Down-with-Imperialism Union (DIU), 오늘의 보도중에서 = From Today's News, 중앙신문개관 = Review of Central Newspapers. "
    "TITLES: 원수님 = Marshal (아버지원수님 = the fatherly Marshal), 장군님 = General, 수령님 = leader; keep the same word for the same Korean throughout. "
    "PLAIN ENGLISH: use the plain everyday word a general viewer knows, never a borrowed foreign or unusual word: trio or group of three, not troika; collective farm, not kolkhoz; also when a dictionary gives the unusual word. "
    "LOANWORDS: the DPRK writes foreign words its own way, mostly from Russian, German, English or Japanese, often with doubled consonants (ㄲ ㄸ ㅃ ㅆ ㅉ). Recognise them by sound, work out the foreign original and use the ordinary English word. NEVER romanize a loanword or a common noun; only personal names, place names and proper names of factories, farms or works are romanized. If a word is unclear, use the most likely everyday English word for the context rather than a romanization. Examples: 세멘트 = cement, 뜨락또르 = tractor, 땅크 = tank, 콤퓨터 = computer, 손전화 = mobile phone, 판형콤퓨터 = tablet, 인터네트 = internet, 홈페지 = website, 텔레비죤 = television, 라지오 = radio, 프로그람 = programme, 쏘프트웨어 = software, 뻐스 = bus, 트롤리뻐스 = trolleybus, 무궤도전차 = trolleybus, 까스/가스 = gas, 깜빠니야 = campaign, 로케트 = rocket, 로보트 = robot, 레이자 = laser, 쎈서 = sensor, 렌트겐 = X-ray, 아빠트/아빠뜨 = apartment block, 살림집 = homes, 메가와트 = megawatt, 키로와트 = kilowatt, 메터 = metre, 키로메터 = kilometre, 센치메터 = centimetre, 그람 = gram, 키로그람 = kilogram, 리터 = litre, 톤 = tonne, 프로/%= per cent, 정보 (land) = chongbo (about a hectare), 꼼바인 = combine harvester, 뽐프 = pump, 모터 = motor, 보이라 = boiler, 베아링 = bearing, 빠이프 = pipe, 레루 = rail, 꼰베아 = conveyor belt, 크레인/기중기 = crane, 다이야 = tyre, 미싱 = sewing machine, 뼁끼 = paint, 비닐 = plastic sheeting, 슬레트 = roofing slate, 타일 = tiles, 콩크리트 = concrete, 아스팔트 = asphalt, 알루미니움 = aluminium, 마그네샤 = magnesia, 마그네사이트 = magnesite, 카바이드 = carbide, 쏘다 = soda, 가성소다 = caustic soda, 비날론 = Vinalon, 그라프 = graph, 쏘세지 = sausage, 쵸콜레트 = chocolate, 고기겹빵 = hamburger, 단묵 = jelly, 사이다 = fizzy drink, 치약 = toothpaste, 뽈 = ball, 꼴 = goal, 롱구 = basketball, 마라손 = marathon, 레스링 = wrestling, 써클 = club, 쎈터 = centre, 로씨야 = Russia, 도이췰란드 = Germany, 웽그리아 = Hungary, 뽈스까 = Poland, 체스꼬 = the Czech Republic, 이딸리아 = Italy, 에스빠냐 = Spain, 메히꼬 = Mexico, 꾸바 = Cuba, 윁남 = Viet Nam, 캄보쟈 = Cambodia, 에짚트 = Egypt, 오스트랄리아 = Australia, 구라파 = Europe, 아메리카 = America. Use plain everyday English, not technical jargon (점토 = clay, 석회석 = limestone, 형석 = fluorite). "
    "Write the English the way the DPRK's own English media (KCNA, The Pyongyang Times, Voice of Korea) do. "
    "NAMES: DPRK romanization, never South Korean spellings. Each syllable of a given name is a separate capitalized word "
    "with NO hyphen: Kim Il Sung, Kim Jong Il, Kim Jong Un, Ri Sol Ju, Kim Yo Jong, Choe Ryong Hae, Pak Thae Song, "
    "Jo Yong Won, Kim Tok Hun, Ri Il Hwan, Choe Son Hui, No Kwang Chol. Surnames: 리/이 = Ri, 로/노 = Ro or No as KCNA writes it, "
    "류/유 = Ryu, 박 = Pak, 조 = Jo, 정 = Jong, 최 = Choe, 강 = Kang, 한 = Han, 문 = Mun. Aspirated consonants are written "
    "Kh, Th, Ph, Ch (Thae, Phyo); ㅓ and ㅗ = o, ㅡ and ㅜ = u, ㅕ = yo. "
    "PLACES as KCNA spells them, one word: Pyongyang, Mt. Paektu (not Baekdu), Samjiyon, Kaesong, Wonsan, Hamhung, Chongjin, "
    "Sinuiju, Rason, Kangwon Province, South Phyongan Province, Taedong River, Kumsusan Palace of the Sun, Mansudae, "
    "Yonggwang County. "
    "TERMS: great leader Comrade Kim Il Sung; great leader Comrade Kim Jong Il; respected Comrade Kim Jong Un; "
    "경애하는 총비서동지 = the respected General Secretary; 어버이 수령님 = the fatherly leader; 장군님 = the General; "
    "조선로동당 = the Workers' Party of Korea (WPK); 당중앙(위원회) = the Party Central Committee; 최고인민회의 = the Supreme People's "
    "Assembly; 국무위원회 = the State Affairs Commission; 조선인민군 = the Korean People's Army (KPA); 조국해방전쟁 = the Fatherland "
    "Liberation War; 항일혁명투쟁 = the anti-Japanese revolutionary struggle; 천리마 = Chollima; 주체 = Juche; 선군 = Songun; "
    "공화국 = the Republic; 조선 = Korea or the DPRK — never 'North Korea' or 'North Korean'. In the names of organizations 조선 is "
    "'Korean': 조선작가동맹 = the Korean Writers Union, 조선중앙통신 = KCNA, 조선중앙텔레비죤 = KCTV, 조선소년단 = the Korean Children's Union. "
    "Keep the respectful register of the original.")

PROMPT = (
    "This is audio from North Korean television (KCTV, Sports TV or Ryongnamsan TV). Most of it is Korean, but some "
    "programmes (often on Ryongnamsan TV) are in English, German, Russian, Chinese (Mandarin) or another language. "
    "List every stretch of SPOKEN or SUNG words, in whatever language, with its time in this clip. Ignore pure music, "
    "tones and silence (return an empty list if nobody speaks or sings). For each stretch give the start and end time in "
    "seconds from the start of the clip (as precise as you can, 0.1 s), in 'ko' the original words exactly as heard in "
    "their own language (Korean in DPRK spelling, e.g. 로동, 리), in 'lang' the language code (ko, en, de, ru, zh, ...), "
    "and in 'en' a natural, concise English subtitle: a translation, or for English speech the words as spoken, tidied "
    "for subtitles. "
    "Include EVERY spoken sentence: never skip, shorten away or summarise speech, and include words cut off at the very "
    "start or end of the clip. "
    "SUBTITLE STYLE for the English: concise and natural, sentence case, numbers as digits, NO full stop or trailing comma "
    "at the end, no filler or hesitation words (예, 애, 음, 네 as filler -> leave out), no speaker labels; each stretch must "
    "fit in a subtitle of at most 64 characters, so split long sentences into separate stretches at natural clause "
    "breaks, each with its own times. "
    "Keep each stretch to one sentence or clause (at most about 7 seconds). Mark kind as 'song' ONLY when the words "
    "are actually SUNG to a melody (a song, a choir, sung lyrics); everything spoken is 'speech' — including narration, "
    "a presenter, poetry recitation or dialogue with music playing underneath. When unsure, use 'speech'. "
    "SONGS: give one stretch per sung lyric line (usually 3-7 seconds), starting exactly when that line starts to be "
    "sung, never two lines merged into one, and include every line of every verse and refrain. " + DPRK_STYLE)
SCHEMA = {"type": "OBJECT", "properties": {"segments": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "start": {"type": "NUMBER"}, "end": {"type": "NUMBER"}, "kind": {"type": "STRING"}, "lang": {"type": "STRING"}, "ko": {"type": "STRING"}, "en": {"type": "STRING"}},
    "required": ["start", "end", "kind", "ko", "en"]}}}, "required": ["segments"]}

# Whisper-style inventions that also turn up on music or silence
HALLUCINATIONS = ('시청해주셔서', '시청해 주셔서', '다음 영상에서', '구독', '좋아요', '알림 설정', '자막 제공', '자막제공', 'MBC 뉴스', '영상편집')


def say(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def run(args):
    r = subprocess.run(args, capture_output=True)
    if r.returncode:
        raise SystemExit('%s failed:\n%s' % (args[0], r.stderr[-1500:].decode(errors='replace')))
    return r


def check_tools():
    for t in ('ffmpeg', 'ffprobe'):
        if not shutil.which(t):
            sys.exit('%s was not found. Install ffmpeg (https://ffmpeg.org/download.html) and make sure ffmpeg and ffprobe are on your PATH.' % t)


def dprk_english(t):
    t = re.sub(r"\bNorth Korean\b", "Korean", t)
    t = re.sub(r"\b[Tt]he North Korea\b|\bNorth Korea\b", "the DPRK", t)
    t = re.sub(r"\b([A-Z][a-z]{1,5}) ([A-Z][a-z]{0,5})-([a-z]{1,5})\b", lambda m: "%s %s %s" % (m.group(1), m.group(2), m.group(3).capitalize()), t)
    return t.replace("the the DPRK", "the DPRK")


SPOKEN_ENDING = re.compile(r'(니다|니까|십시오|십시요|세요|ㅂ시다|읍시다|습니까|겠습니다)[\s.?!…"\'」』)]*$')


def is_song(kind, ko, en):
    """Stricter than Gemini's label (songs get no subtitles): narration, news and commentary end in formal endings
    (-습니다/-ㅂ니다), lyrics almost never do; a long line is narration too."""
    if not str(kind or '').lower().startswith('song'):
        return False
    return not SPOKEN_ENDING.search(str(ko or '').strip()) and len(en) <= 92


def snap_to_speech(lines, x):
    """Move each line's start/end to where speech actually starts/stops nearby (Gemini's times are rough)."""
    if not lines or len(x) < SR:
        return lines
    hop = 320
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop).astype(np.float32)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    act = db > np.percentile(db, 20) + 10
    act = np.convolve(act.astype(np.int32), np.ones(10, dtype=np.int32), 'same') > 5
    on = (np.where(~act[:-1] & act[1:])[0] + 1) * 0.02
    off = (np.where(act[:-1] & ~act[1:])[0] + 1) * 0.02
    out = []
    for st, et, en in lines:
        c = on[(on >= st - 1.0) & (on <= st + 0.6)]
        st2 = float(c[np.argmin(np.abs(c - st))]) if len(c) else st
        c = off[(off >= et - 0.6) & (off <= et + 1.0) & (off > st2 + 0.6)]
        et2 = float(c[np.argmin(np.abs(c - et))]) if len(c) else et
        out.append((st2, max(et2, st2 + 0.8), en))
    return out


try:
    DICTIONARY = set(w.strip().lower() for w in open('/usr/share/dict/words'))
except Exception:
    DICTIONARY = set()


LOANWORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'loanwords.json')   # optional, next to this script
LANG_NAMES = {'영': 'English', '로': 'Russian', '도': 'German', '라': 'Latin', '프': 'French', '이': 'Italian', '그': 'Greek',
              '일': 'Japanese', '네': 'Dutch', '에': 'Spanish', '헤': 'Hebrew', '중': 'Chinese', '싼': 'Sanskrit', '뽀': 'Portuguese'}


def load_loanwords():
    """DPRK loanwords from the DPRK's own dictionaries (Samhung: the Korean dictionary marks each loanword with its origin,
    e.g. 세멘트 [cement《영》], 뜨락또르 [трактор《로》]; the Korean-English one gives the meaning). Built by
    /DATA/subs/dict/build_loanwords.py into {word: {en, from, lang}}."""
    try:
        table = json.load(open(LOANWORD_FILE))
    except Exception as e:
        return {}, []
    return table, sorted(table, key=len, reverse=True)


LOANWORDS, LOANWORD_KEYS = load_loanwords()


def loanwords_in(original):
    """Loanwords in a Korean line, longest first, not overlapping (Korean attaches particles and compounds to them)."""
    found, taken = [], [False] * len(original)
    if not LOANWORD_KEYS or not re.search('[가-힣]', original):
        return found
    for w in LOANWORD_KEYS:
        i = original.find(w)
        while i >= 0:
            if not any(taken[i:i + len(w)]):
                for k in range(i, i + len(w)): taken[k] = True
                found.append(w); break
            i = original.find(w, i + 1)
    return found


def loan_hint(w):
    e = LOANWORDS[w]
    src = '%s (%s)' % (e['from'], LANG_NAMES.get(e['lang'], e['lang']))
    return '%s = %s' % (w, e['en'] + ', from ' + src if e.get('en') else 'from ' + src)


def loan_missing(w, en):
    """True when the English shows no sign of the loanword's meaning."""
    e = LOANWORDS[w]
    cues = [x for x in re.findall(r'[A-Za-z]{4,}', (e.get('en') or '') + ' ' + (e['from'] if e['lang'] == '영' else ''))]
    if not cues:
        return False                                   # nothing to check against: leave Gemini's choice alone
    low = en.lower()
    return not any(c.lower()[:max(4, len(c) - 3)] in low for c in cues)


def odd_words(en, any_case=False):
    """Words the dictionary does not know: a romanized loanword ("saeminto"). Lower case only unless any_case
    (then names count too: "Saeminto Factory" was a loanword in a name)."""
    if not DICTIONARY:
        return []
    return [w for w in re.findall(r"\b[A-Za-z][a-z]+\b" if any_case else r"\b[a-z][a-z]+\b", en)
            if not any(x in DICTIONARY for x in (w.lower(), w.lower()[:-1], w.lower()[:-2], w.lower()[:-3]))]


def fix_loanwords(key, lines):
    """Lines whose Korean uses a dictionary loanword the English does not reflect (세멘트 came out as "saeminto") go
    back to the same model as text, with the dictionary entries. Small and only when needed; any failure keeps
    the first answer."""
    todo = []
    for i, (st, et, en, ko, kind) in enumerate(lines):
        words = loanwords_in(ko)
        # only when the English has a word that is not English ("saeminto"): letters of a loanword can sit
        # inside an ordinary Korean word, and a "fix" then broke a correct line ("match" -> "gear")
        if words and odd_words(en, any_case=True):
            todo.append((i, [loan_hint(w) for w in words]))
    if not todo:
        return
    items = [{'n': k + 1, 'korean': lines[i][3], 'english': lines[i][2], 'dictionary': hints} for k, (i, hints) in enumerate(todo)]
    prompt = ("These English subtitles were translated from DPRK television. Some Korean words are DPRK loanwords; "
              "the DPRK dictionary entries for them are given. Where the English romanized a loanword, mistranslated "
              "it or left it out, correct the English using the dictionary meaning (choose the sense that fits the "
              "context; a dictionary entry often does not apply, because the same letters can be part of an ordinary Korean "
              "word: if the English already says it correctly, return it EXACTLY unchanged). Change "
              "nothing else and keep the subtitle style (concise, no full stop at the end). " + DPRK_STYLE +
              ' Return {"lines": [...]} with one English string per item, in order.\n' + json.dumps(items, ensure_ascii=False))
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                            "responseSchema": {"type": "OBJECT", "properties": {"lines": {"type": "ARRAY", "items": {"type": "STRING"}}},
                                                               "required": ["lines"]}}}).encode()
    req = gemini_request(key, MODEL, body)
    try:
        r = json.load(urllib.request.urlopen(req, timeout=40))
        fixed = json.loads(r['candidates'][0]['content']['parts'][0]['text'])['lines']
        if len(fixed) != len(todo):
            return
        for (i, hints), en in zip(todo, fixed):
            en = str(en or '').strip()
            if en and en != lines[i][2]:
                lines[i][2] = en
    except Exception as e:
        pass


def bad_original(t, lang):
    """Korean lines get the Korean checks; another language (English, German, Russian, Chinese programmes) is only
    dropped when empty or a single repeated character."""
    if str(lang or 'ko').lower().startswith('ko') or re.search('[\uac00-\ud7a3]', t):
        return bad_korean(t)
    return len(set(re.sub(r'[\s.,!?~…·-]', '', t))) <= 1


def bad_korean(ko):
    core = re.sub(r'[\s.,!?~…·-]', '', ko)
    return not core or len(core) <= 2 or len(set(core)) <= 2 or any(h in ko for h in HALLUCINATIONS)


# ------------------------------------------------------------------ sound analysis
def sound_kind(x):
    """'tone' (an off-air test signal: one to three pure tones at a constant level), 'silence' or 'sound'."""
    if len(x) < 1600:
        return 'silence'
    rms = 20 * np.log10(np.sqrt(float((x ** 2).mean())) + 1e-9)
    if rms < -50:
        return 'silence'
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    total, got, Y = float(X.sum()) + 1e-12, 0.0, X.copy()
    for _ in range(3):
        k = int(Y.argmax()); band = np.abs(f - f[k]) < 25
        got += float(Y[band].sum()); Y[band] = 0
    q = len(x) // 4
    levels = [20 * np.log10(np.sqrt(float((x[i * q:(i + 1) * q] ** 2).mean())) + 1e-9) for i in range(4)]
    return 'tone' if got / total > 0.90 and max(levels) - min(levels) < 0.5 else 'sound'


def decode_audio(path, out):
    # locked to the sound's own timestamps: a broadcast's audio clock can run fast and lose frames, and counted end to
    # end the sound drifts from the picture (0.15-0.7 s after 13 h of KCTV)
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', path, '-map', '0:a:0', '-ac', '1',
         '-af', 'aresample=%d:async=1:min_hard_comp=0.02:first_pts=0' % SR, '-f', 's16le', out])
    return np.memmap(out, dtype=np.int16, mode='r')


def kinds(a):
    return [sound_kind(a[i * SR:(i + 1) * SR].astype(np.float32) / 32768) for i in range(len(a) // SR)]


# ------------------------------------------------------------------ Gemini
def gemini(key, flac, secs, tries=20):
    body = json.dumps({"contents": [{"parts": [{"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(flac).decode()}}, {"text": PROMPT}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA}}).encode()
    for attempt in range(tries):
        model = MODEL                                              # one model only; busy -> wait and try again
        req = gemini_request(key, model, body)
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120 + secs))
            lines = []
            for g in json.loads(r['candidates'][0]['content']['parts'][0]['text'])['segments']:
                en, ko = str(g.get('en') or '').strip(), str(g.get('ko') or '').strip()
                st, et = float(g.get('start', 0)), float(g.get('end', 0))
                if not en or bad_original(ko, g.get('lang')):
                    continue
                st = min(max(0.0, st), secs); et = min(max(et, st + 1.2), secs + 2)
                lines.append([st, et, en, ko, g.get('kind')])
            fix_loanwords(key, lines)                                   # DPRK loanwords checked against the dictionary
            out = []
            for st, et, en, ko, kind in lines:
                en = dprk_english(en)
                out.append((st, et, ('♪ ' + en) if is_song(kind, ko, en) else en))
            return out
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403):
                raise SystemExit('Gemini refused the request (HTTP %d): check the API key.\n%s' % (e.code, e.read()[:400].decode(errors='replace')))
            time.sleep(min(60, 4 * 2 ** min(attempt, 4)))      # rate limit or busy: wait and try again
        except Exception:
            time.sleep(5)
    return None


def to_flac(x):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    return subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 's16le', '-ar', str(SR), '-ac', '1', '-i', '-', '-c:a', 'flac', '-f', 'flac', '-'],
                          input=pcm, capture_output=True).stdout


# ------------------------------------------------------------------ subtitle tidying
def tidy_lines(lines):
    # no subtitles for songs: Gemini times sung lyrics badly (late, or missing). A long "song" line is narration or
    # recitation over music that was mislabelled, so it is kept as ordinary speech.
    lines = [(a, b, t[2:] if t.startswith('♪ ') else t) for a, b, t in lines if not t.startswith('♪ ') or len(t) > 92]   # long "song" line = narration
    lines = sorted(lines, key=lambda x: (x[0], x[1]))
    out, i = [], 0
    while i < len(lines):                                     # lines that start together share their span, by length
        j = i
        while j + 1 < len(lines) and abs(lines[j + 1][0] - lines[i][0]) < 0.05:
            j += 1
        grp = lines[i:j + 1]
        s0, e0 = grp[0][0], max(x[1] for x in grp)
        tot = sum(len(x[2]) for x in grp) or 1
        t = s0
        for x in grp:
            d = (e0 - s0) * len(x[2]) / tot
            out.append([t, t + d, x[2]]); t += d
        i = j + 1
    for k in range(len(out) - 1):                             # no overlaps: each line ends where the next begins
        if out[k][1] > out[k + 1][0]:
            out[k][1] = max(out[k][0] + 0.5, out[k + 1][0])
    return out


LINE = 32                                                    # characters per line, 2 lines per subtitle: the CEA-608
                                                             # caption row is 32 wide, so WebVTT and CC break the same way
WEAK_END = {'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'or', 'but', 'with', 'by', 'from', 'that', 'his',
            'her', 'their', 'our', 'its', 'my', 'your', 'as', 'is', 'are', 'was', 'were', 'be', 'will', 'has', 'have', 'who', 'which'}
BREAK_BEFORE = {'and', 'but', 'or', 'that', 'which', 'who', 'where', 'when', 'while', 'because', 'to', 'with', 'for', 'as'}


FILLER = re.compile(r",?\s*(?<![\w'])(?:u+h+|u+m+|e+r+[mn]*|a+h+|e+h+|h+m+|m{2,}|h+u+h+|uh-huh)(?![\w'])[,.]?", re.I)


LEADERS = re.compile(r"\bKim(\s+)(Il(\s+)Sung|Jong(\s+)Il|Jong(\s+)Un|Jong(\s+)Suk)\b")


def bold_leaders(t):
    """The leaders' names in bold (user's request): Kim Il Sung, Kim Jong Il, Kim Jong Un, Kim Jong Suk. A name split
    over the two lines of a subtitle is bolded on each line."""
    return LEADERS.sub(lambda m: '\n'.join('<b>%s</b>' % part for part in m.group(0).split('\n')), t)


def clean_text(t):
    """Subtitle text: no end punctuation (? and ! keep their meaning), no ellipses or filler, single spaces."""
    t = re.sub(r'\s+', ' ', str(t)).strip()
    t = t.replace('…', ' ').replace('...', ' ')
    t = re.sub(r'^(Yes|Uh|Um|Well|Ah|Oh),\s+', '', t)                     # filler the translator kept
    t = FILLER.sub(' ', t)                                                # hesitation sounds anywhere ("ern", "erm", "uh")
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.I)                 # stumbles: "this this", "from from"
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'[\s.,;:\-–—]+$', '', t)                                 # no full stop / trailing comma at the end
    return t[:1].upper() + t[1:] if t else t


def split_two(text):
    """One subtitle's text as 1 or 2 balanced lines of at most LINE characters (None if it does not fit)."""
    if len(text) <= LINE:
        return [text]
    words, best = text.split(' '), None
    for i in range(1, len(words)):
        a, b = ' '.join(words[:i]), ' '.join(words[i:])
        if len(a) > LINE or len(b) > LINE:
            continue
        score = abs(len(a) - len(b))
        last = words[i - 1].lower().strip(',;:')
        if words[i - 1].endswith((',', ';', ':')): score -= 14            # break after a comma
        if words[i].lower() in BREAK_BEFORE: score -= 8                     # break before a conjunction
        if last in WEAK_END: score += 25                                    # never leave "the"/"of" hanging
        if i == 1 or i == len(words) - 1: score += 30                       # no line of a single word
        if best is None or score < best[0]:
            best = (score, [a, b])
    return best[1] if best else None


def split_cues(text):
    """Text too long for one subtitle -> several, each at most 2 x LINE, cut at clause boundaries where possible."""
    if split_two(text):
        return [text]
    words, out, cur = text.split(' '), [], []
    for w in words:
        cand = ' '.join(cur + [w])
        if cur and not split_two(cand):
            # step back to the last comma/conjunction inside cur, if that keeps a decent first part
            cut = len(cur)
            for k in range(len(cur) - 1, max(0, len(cur) - 6), -1):
                if cur[k - 1].endswith((',', ';', ':')) or cur[k].lower() in BREAK_BEFORE:
                    if len(' '.join(cur[:k])) >= 30: cut = k; break
            out.append(' '.join(cur[:cut])); cur = cur[cut:] + [w]
        else:
            cur.append(w)
    if cur: out.append(' '.join(cur))
    # a short tail ("people", "at the farm") is not a subtitle on its own: share the last two parts evenly
    if len(out) >= 2 and (len(out[-1]) < 22 or ' ' not in out[-1]):
        both = (out[-2] + ' ' + out[-1]).split(' ')
        best = None
        for i in range(1, len(both)):
            a, b = ' '.join(both[:i]), ' '.join(both[i:])
            if split_two(a) and split_two(b):
                score = abs(len(a) - len(b)) - (12 if both[i - 1].endswith((',', ';', ':')) else 0)
                if best is None or score < best[0]:
                    best = (score, a, b)
        if best:
            out[-2:] = [best[1], best[2]]
    return out


def make_cues(lines):
    """Lines (start, end, english) -> finished subtitles (start, end, 'line1\nline2') following subtitle rules:
    songs dropped and overlaps tidied (tidy_lines), clean text, at most 2 balanced lines of 42, long lines split
    at clause breaks with time shared by length, at least 1 s / ~17 characters per second on screen (stretching
    into free time), at most 7 s, and an 80 ms gap so a change of subtitle is visible."""
    cues = []
    for st, et, en in tidy_lines(lines):
        # capital only at the start of the sentence; a continuation keeps its lower case
        parts = [re.sub(r'[\s.,;:\-–—]+$', '', p).strip() for p in split_cues(clean_text(en))]
        parts = [p for p in parts if re.search(r'[A-Za-z0-9]', p)]
        if not parts:
            continue
        tot = sum(len(p) for p in parts) or 1
        t = st
        for p in parts:
            d = (et - st) * len(p) / tot
            cues.append([t, t + d, p]); t += d
    cues.sort(key=lambda c: c[0])
    for i, c in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else c[0] + 7.0
        want = max(1.0, len(c[2]) / 17.0)
        c[1] = max(c[1], min(c[0] + want, nxt - 0.08))                     # long enough to read, if there is room
        c[1] = min(c[1], c[0] + 7.0, nxt - 0.08 if i + 1 < len(cues) else c[1])
        c[1] = max(c[1], c[0] + 0.5)
    return [(a, b, '\n'.join(split_two(t) or [t])) for a, b, t in cues]


def srt_time(s):
    ms = int(round(max(0, s) * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); sec, ms = divmod(ms, 1000)
    return '%02d:%02d:%02d,%03d' % (h, m, sec, ms)


def count_packets(path):
    r = run(['ffprobe', '-v', 'error', '-count_packets', '-show_entries', 'stream=codec_type,nb_read_packets', '-of', 'json', path])
    st = json.loads(r.stdout)['streams']
    return {k: sum(int(s.get('nb_read_packets') or 0) for s in st if s['codec_type'] == k) for k in ('video', 'audio', 'subtitle')}


# ------------------------------------------------------------------ main
# ---------------------------------------------------------------- the full live method, when its files are here
# subtitles.py (the live service's engine) and data/search.db sit next to this script in the project: it then makes
# subtitles exactly as the live service does
# (house style, DPRK names and place names, the Samhung dictionary looked up by Gemini, songs as plain lines, timed on the singer).
# Without them it uses the built-in method above.
HERE = os.path.dirname(os.path.abspath(__file__))
_DB, _XZ = os.path.join(HERE, 'data', 'search.db'), os.path.join(HERE, 'data', 'search.db.xz')
if not os.path.exists(_DB) and os.path.exists(_XZ):          # the repository ships the database compressed
    import lzma
    print('unpacking the dictionary database (once) ...')
    with lzma.open(_XZ) as fi, open(_DB + '.part', 'wb') as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    os.replace(_DB + '.part', _DB)
LIVE = None
try:
    if os.path.exists(os.path.join(HERE, 'subtitles.py')):
        for db in (os.path.join(HERE, 'data', 'search.db'), os.path.join(HERE, 'search.db')):
            if os.path.exists(db):
                os.environ.setdefault('SUBS_DICT_DB', db)    # dictionaries, place names and loanwords: one database
                break
        os.environ.setdefault('SUBS_STATE', os.path.join(HERE, 'state'))
        os.makedirs(os.environ['SUBS_STATE'], exist_ok=True)
        sys.path.insert(0, HERE)
        import subtitles as LIVE
        LIVE.PLACES, LIVE.PLACE_KEYS = LIVE.load_places()
except Exception as e:                                     # an old or broken copy: keep the built-in method
    print('(not using subtitles.py:', type(e).__name__, str(e)[:80], ')')
    LIVE = None
_live_gemini = {}


def live_gemini(key, flac, secs, tries=20, x=None):
    g = _live_gemini.setdefault(key, LIVE.Gemini(key))
    expect = x is not None and LIVE.voiced_seconds(x, 0, secs) > 0.35 * secs
    for attempt in range(max(1, tries // 5)):
        out = g.subtitles(flac, secs, time.time() + 300, expect)
        if out is not None:
            return out
        time.sleep(10)
    return None


def main():
    check_tools()
    print(__doc__.strip().split('\n\n')[0], '\n')
    src = input('Path of the recording: ').strip().strip('"').strip("'")
    if not os.path.isfile(src):
        sys.exit('No such file: %s' % src)
    key = (os.environ.get('GEMINI_API_KEY') or
           getpass.getpass('Gemini API key - Google AI Studio, or Vertex AI (starts AQ.) (not shown as you type): ').strip())
    if not key:
        sys.exit('An API key is needed (aistudio.google.com -> Get API key).')
    trim = input('Trim to the broadcast (drop the off-air test signals before and after)? [Y/n]: ').strip().lower() not in ('n', 'no')

    base = os.path.splitext(src)[0]
    W = base + '_subtitle-work' + os.sep
    os.makedirs(W + 'chunks', exist_ok=True)
    straight, cut, raw = W + 'straight.ts', W + 'cut.ts', W + 'sound.raw'
    srt, mkv = base + '_English.srt', base + '_English.mkv'

    # 1. lossless timestamp straightening (also makes a .mkv/.mp4 input a plain TS for the steps below)
    if not os.path.exists(cut) and not os.path.exists(straight):
        say('straightening timestamps (stream copy, nothing re-encoded) ...')
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-fflags', '+genpts', '-i', src, '-map', '0:v:0', '-map', '0:a:0', '-c', 'copy',
             '-avoid_negative_ts', 'make_zero', '-muxdelay', '0', '-muxpreload', '0', '-f', 'mpegts', straight + '.part'])
        os.replace(straight + '.part', straight)

    # 2. find the broadcast and cut there
    if not os.path.exists(cut):
        if trim:
            say('finding the start and end of the broadcast from the sound ...')
            a = decode_audio(straight, W + 'full.raw')
            k = kinds(a)
            snd = np.array([x == 'sound' for x in k], dtype=np.int8)
            win = np.convolve(snd, np.ones(30, dtype=np.int32), 'valid')
            good = np.where(win >= 27)[0]
            if len(good) == 0:
                sys.exit('No programme sound found in this recording.')
            start = int(good[0])
            while start > 0 and k[start - 1] != 'tone':      # back to where the test signal stops (sign-on included)
                start -= 1
            end, i = int(good[-1] + 30), int(good[-1] + 30)
            while i < len(k) and k[i] != 'tone':              # forward to the last real sound before the off-air tone
                if k[i] == 'sound': end = i + 1
                i += 1
            del a; os.remove(W + 'full.raw')
            say('broadcast from %s to %s of the recording (%.2f h)' % (srt_time(start)[:8], srt_time(end)[:8], (end - start) / 3600))
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', str(max(0, start - 2)), '-i', straight, '-t', str(end - start + 4),
                 '-map', '0:v:0', '-map', '0:a:0', '-c', 'copy', '-avoid_negative_ts', 'make_zero', '-muxdelay', '0', '-muxpreload', '0',
                 '-f', 'mpegts', cut + '.part'])
            os.replace(cut + '.part', cut)
            os.remove(straight)
        else:
            os.replace(straight, cut)

    # 3. subtitles
    if not os.path.exists(srt):
        info = json.loads(run(['ffprobe', '-v', 'error', '-show_entries', 'format=start_time:stream=codec_type,start_time', '-of', 'json', cut]).stdout)
        a_off = float(next(s['start_time'] for s in info['streams'] if s['codec_type'] == 'audio')) - float(info['format']['start_time'])
        say('reading the sound ...')
        a = np.memmap(raw, dtype=np.int16, mode='r') if os.path.exists(raw) else decode_audio(cut, raw)
        k = kinds(a)
        chunks, i, n = [], 0, len(k)
        while i < n:                                          # runs of real sound, split near every 2 minutes at a quiet second
            if k[i] != 'sound':
                i += 1; continue
            j = i
            while j < n and k[j] == 'sound' and j - i < CHUNK_S + 8:
                j += 1
            if j - i > CHUNK_S and j < n and k[j] == 'sound':
                lo = i + CHUNK_S - 8
                rms = [float(np.sqrt(np.mean((a[s * SR:(s + 1) * SR].astype(np.float32) / 32768) ** 2))) for s in range(lo, j)]
                j = lo + int(np.argmin(rms)) + 1
            chunks.append((i, j)); i = j
        say('%d chunks, %.1f h of sound; sending to Gemini (%d at a time) ...' % (len(chunks), sum(b - c for c, b in chunks) / 3600, PARALLEL))
        lock, done = threading.Lock(), [0]

        def one(job):
            ci, (c0, c1) = job
            f = W + 'chunks/%05d.json' % ci
            if os.path.exists(f):
                out = json.load(open(f, encoding='utf-8'))
            else:
                xc = a[c0 * SR:c1 * SR].astype(np.float32) / 32768
                out = live_gemini(key, to_flac(xc), float(c1 - c0), x=xc) if LIVE else gemini(key, to_flac(xc), float(c1 - c0))
                if out is None:
                    return None
                with open(f + '.tmp', 'w', encoding='utf-8') as fh:
                    json.dump(out, fh, ensure_ascii=False)
                os.replace(f + '.tmp', f)
            xc = a[c0 * SR:c1 * SR].astype(np.float32) / 32768
            if LIVE and not out:
                # empty but full of sound: music, or speech Gemini skipped -> transcribed again (as live and as dayjob)
                out = LIVE.recheck_empty(key, xc, float(c1 - c0), _live_gemini.get(key)) or []
                if out:
                    with open(f + '.tmp', 'w', encoding='utf-8') as fh:
                        json.dump(out, fh, ensure_ascii=False)
                    os.replace(f + '.tmp', f)
            with lock:
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == len(chunks):
                    say('  %d/%d chunks done' % (done[0], len(chunks)))
            if LIVE and out:
                # snapped to the voice; a chunk Gemini timed badly is asked again and the best-aligned answer kept
                g = _live_gemini.setdefault(key, LIVE.Gemini(key))
                out = LIVE.well_aligned(g, to_flac(xc), xc, float(c1 - c0), time.time() + 300, True, [tuple(x) for x in out])
                if LIVE.speech_gaps(out, xc):          # speech Gemini left without lines: asked again for those stretches
                    out = LIVE.fill_gaps(g, to_flac(xc), xc, float(c1 - c0), time.time() + 300, True, out)
                    with open(f + '.tmp', 'w', encoding='utf-8') as fh:
                        json.dump(out, fh, ensure_ascii=False)
                    os.replace(f + '.tmp', f)
            else:
                out = snap_to_speech([tuple(x) for x in out], xc)
            if LIVE and out:                                  # the timing (subtitles.timed_lines)
                g = _live_gemini.setdefault(key, LIVE.Gemini(key))
                out = LIVE.song_check(key, xc, [tuple(l[:3]) for l in out])   # 'songs' that are speech or not there
                if any(l[2].startswith('♪') for l in out):
                    # a music video writes its lyrics on screen (lower half): those words, verse by verse as they appear
                    fc = W + 'chunks/%05d.captions.json' % ci
                    cap = json.load(open(fc, encoding='utf-8')) if os.path.exists(fc) else None
                    if cap is None:
                        fr = LIVE.read_captions(key, cut, c0 + a_off - 2, c1 + a_off + 10)
                        if fr is not None:
                            V = LIVE.caption_verses([(t - a_off - c0, q) for t, q in fr])
                            en = LIVE.translate_lyrics(key, [r for v in V for r in v[2]]) if V else []
                            if en is not None:
                                cap = {'verses': V, 'english': en}
                                with open(fc + '.tmp', 'w', encoding='utf-8') as fh:
                                    json.dump(cap, fh, ensure_ascii=False)
                                os.replace(fc + '.tmp', fc)
                    if cap and cap['verses']:
                        out = LIVE.lyrics_from_captions(out, [tuple(v) for v in cap['verses']], cap['english'], 0.0, float(c1 - c0))
                    elif cap is not None:
                        # no lyrics on screen over a still picture (a test card): instrumental music, invented lyrics
                        m = LIVE.picture_motion(cut, c0 + a_off, c1 + a_off)
                        if m is not None and m < LIVE.CAPTION_STILL:
                            out = [l for l in out if not l[2].startswith('♪')]
                out = LIVE.snap_to_speech(out, xc, None, g.korean)
            if LIVE:
                with lock:
                    starts.append(c0 + LIVE.voice_starts(xc))   # where the voice resumes: long lines break there
            return [(c0 + st, c0 + et, en) for st, et, en in out]

        jobs, starts = list(enumerate(chunks)), []
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            results = list(pool.map(one, jobs))
        missing = [j for j, r in zip(jobs, results) if r is None]
        lines = [x for r in results if r for x in r]
        for j in missing:                                     # stubborn chunks: once more, one at a time
            r = one(j)
            if r: lines += r
            else: say('  chunk %d has no subtitles (Gemini kept failing)' % j[0])
        if LIVE:                                              # exactly the live engine's rules, then broadcast shortening
            T = lambda v: v + a_off                        # the decoded sound is on the recording's own clock
            lines = LIVE.fix_foreign(key, sorted(lines))   # a word in the wrong script (garbled Korean) -> English
            cues = LIVE.make_cues([(T(s), T(e), LIVE.dprk_english(t)) for s, e, t in lines],
                                  T(np.sort(np.concatenate(starts))) if starts else None)
            say('shortening subtitles too dense to read ...')
            cues = LIVE.condense_cues(key, cues)
            bold = LIVE.bold_leaders
        else:
            cues = make_cues([(s + a_off, e + a_off, dprk_english(t)) for s, e, t in lines])
            bold = bold_leaders
        with open(srt + '.part', 'w', encoding='utf-8') as fh:
            for i, (st, et, t) in enumerate(cues, 1):
                fh.write('%d\n%s --> %s\n%s\n\n' % (i, srt_time(st), srt_time(et), bold(t)))
        os.replace(srt + '.part', srt)
        del a
        say('%d subtitles written to %s' % (len(cues), srt))

    # 4. MKV: picture and sound copied untouched, plus the English track; then check every packet is there
    say('building the MKV ...')
    a = count_packets(cut)
    # mkvmerge (MKVToolNix) when installed: the reference Matroska muxer, safest for interlaced broadcast picture
    if shutil.which('mkvmerge'):
        r = subprocess.run(['mkvmerge', '-q', '-o', mkv + '.part.mkv', '--title', '%s (English subtitles, machine translated)'
                            % os.path.basename(base), cut, '--language', '0:eng', '--track-name', '0:English',
                            '--default-track-flag', '0:yes', srt])
        b = count_packets(mkv + '.part.mkv') if r.returncode in (0, 1) else None
        if b and (a['video'], a['audio']) == (b['video'], b['audio']):
            os.replace(mkv + '.part.mkv', mkv)
            shutil.rmtree(W, ignore_errors=True)
            say('done (every picture and sound packet checked):\n  %s\n  %s' % (mkv, srt))
            return
        say('mkvmerge result refused, building with ffmpeg instead ...')
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-fflags', '+genpts', '-i', cut, '-i', srt, '-map', '0:v:0', '-map', '0:a:0', '-map', '1:0',
         '-c', 'copy', '-c:s', 'srt',
         '-bsf:v', 'setts=pts=if(eq(PTS\\,NOPTS)\\,PREV_OUTPTS+3600\\,PTS):dts=if(eq(DTS\\,NOPTS)\\,PREV_OUTDTS+3600\\,DTS)',
         '-metadata:s:s:0', 'language=eng', '-metadata:s:s:0', 'title=English', '-disposition:s:0', 'default',
         '-metadata', 'title=%s (English subtitles, machine translated)' % os.path.basename(base), mkv + '.part.mkv'])
    say('checking that every picture and sound packet is in the MKV ...')
    b = count_packets(mkv + '.part.mkv')
    if (a['video'], a['audio']) != (b['video'], b['audio']):
        sys.exit('Check failed: source %s, MKV %s. The work folder is kept: %s' % (a, b, W))
    os.replace(mkv + '.part.mkv', mkv)
    shutil.rmtree(W, ignore_errors=True)
    say('done:\n  %s\n  %s' % (mkv, srt))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Run it again to carry on where it left off.')
