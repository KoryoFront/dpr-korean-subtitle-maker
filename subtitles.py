#!/usr/bin/env python3
"""Live English subtitles for KCTV, Sports TV and Ryongnamsan TV.

Runs beside the encoders and touches nothing of theirs: it reads each channel's
finished 6 s HLS segments from disk, and publishes subtitles next to the video
on R2/Cloudflare:

    <prefix>/master.m3u8        video + an "English" subtitle track (what a player opens)
    <prefix>/index-sub.m3u8     the same video segments, held back until their subtitles exist
    <prefix>/subs/en.m3u8       subtitle playlist, one WebVTT per video segment
    <prefix>/subs/en<seq>.vtt   the subtitles for video segment <seq>

The existing <prefix>/index.m3u8 is not touched, so current players are unaffected.

On air or not is decided by the SOUND, not the clock: the channels play a pure
1 kHz tone when off air (and sometimes silence). Those segments are never sent
anywhere and cost nothing; everything else is batched (~18 s) to Groq Whisper
large-v3 (Korean), cleaned of the usual Whisper hallucinations, translated to
English by a Groq text model, and timed to the video with X-TIMESTAMP-MAP.

Config: /etc/stream-gui/subtitles.json  {"groq_key": "..."} (root, 0600)
        /etc/stream-server.env          R2_* (the uploader's own credentials)
"""
import base64, datetime, heapq, hashlib, hmac, http.client, json, os, re, subprocess, sys, threading, time, urllib.parse, urllib.request, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import cea608                                                  # captions inside the video (switched off: CC_ENABLED)
except ImportError:
    cea608 = None

CHANNELS = [
    {'key': 'kctv',        'dir': '/var/www/stream',             'prefix': 'kctv'},
    {'key': 'sports',      'dir': '/var/www/stream/sports',      'prefix': 'sports'},
    {'key': 'ryongnamsan', 'dir': '/var/www/stream/ryongnamsan', 'prefix': 'ryongnamsan'},
]
CONFIG = os.environ.get('SUBS_CONFIG', '/etc/stream-gui/subtitles.json')
ENV_FILE = os.environ.get('SUBS_ENV', '/etc/stream-server.env')
STATE_DIR = os.environ.get('SUBS_STATE', '/DATA/subs')            # never /tmp: it is RAM on this server
DRY = '--dry' in sys.argv                                          # write files under STATE_DIR/dry instead of R2

# First choice: Gemini Flash-Lite (free tier) transcribes AND translates in one call. In tests it beat Whisper on
# music (no invented lines) and caught speech Whisper replaced with a fake credit. Backup when Gemini is busy or
# rate-limited: Groq Whisper large-v3 + a Groq text model (also free tier, ~8 h of audio a day).
GEMINI_MODEL = 'gemini-flash-lite-latest'
PRICE_IN, PRICE_OUT = 0.30, 2.50                                   # USD per million tokens, Gemini 3.5 Flash-Lite (what 'latest' is now), audio in at the same rate
USAGE_FILE = os.path.join(os.environ.get('SUBS_STATE', '/DATA/subs'), 'usage.json')
GEMINI_BACKUP = None                                               # the user wants one model only: busy -> retry the same one
SPEECH_MODEL = 'whisper-large-v3'                                  # turbo dropped whole stretches of Korean speech in tests
TEXT_MODEL = 'qwen/qwen3.8-27b'                                    # best names/places in tests, and cheapest per line
BATCH_S = 30.0                                                     # audio per request; viewers are held back HOLD_SEGMENTS (src/r2/held.js)
HOLD_SEGMENTS = 18                                                 # must match src/r2/held.js: 108 s
WINDOW = 30                                                        # segments in subs/en.m3u8: must reach back past the held-back
                                                                   # picture (18 segments + video.m3u8 window 8), or players find no subtitles
MAX_HOURS_PER_DAY = 30.0                                           # spend guard: never send more audio than this per day
RETENTION_S = 900                                                  # subtitle files live as long as the segments (R2_RETENTION_MS)
REQUEST_TIMEOUT = 60                                               # s for one Gemini answer (normally 5-15 s, much slower when Google is busy)
BATCH_DEADLINE = 100                                               # s after a batch's first segment appeared (viewers reach it at 108 s)
TERM_CHECK_S = 12                                                  # most time the dictionary check may take per batch
CC_ENABLED = False                                                 # the user wants WebVTT only: no captions inside the video
CC_DEADLINE = 75                                                   # s after a segment appears: caption it even if the next
                                                                   # segment's subtitles are not known yet (viewers reach it at 108 s)
TICK = 90000                                                       # MPEG-TS clock, Hz
WRAP = 1 << 33

TONE_SHARE = 0.90                                                  # test signals: ~100 % of energy in their 1-3 tones; speech/music 10-60 %, never steady
SILENCE_DB = -50.0

# Whisper's well-known inventions on music, applause or silence (Korean YouTube subtitles in its training data).
HALLUCINATIONS = ('시청해주셔서', '시청해 주셔서', '다음 영상에서', '구독', '좋아요', '알림 설정', '자막 제공', '자막제공', 'MBC 뉴스', '영상편집',
                  'thank you for watching', 'thanks for watching', 'subscribe')

SYSTEM_PROMPT = None   # set below, once DPRK_STYLE exists


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def load_env(path):
    env = {}
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*([A-Z0-9_]+)=(.*)$', line)
        if m:
            v = m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
                v = v[1:-1]
            env[m.group(1)] = v
    return env


# ---------------------------------------------------------------- R2 (S3 SigV4, no extra processes)
class R2:
    def __init__(self, env):
        self.ep = urllib.parse.urlparse(env['R2_ENDPOINT'])
        self.bucket = env['R2_BUCKET']
        self.ak, self.sk = env['R2_ACCESS_KEY_ID'], env['R2_SECRET_ACCESS_KEY']
        self.public = env['R2_PUBLIC_BASE'].rstrip('/')
        self.lock = threading.Lock()
        self.conn = None
        self.expiry = []                                             # (due time, key) of uploaded subtitle files

    def _sign(self, method, path, payload_hash, headers):
        now = datetime.datetime.now(datetime.timezone.utc)
        amzdate, date = now.strftime('%Y%m%dT%H%M%SZ'), now.strftime('%Y%m%d')
        headers = {k.lower(): str(v).strip() for k, v in headers.items()}
        headers.update({'host': self.ep.netloc, 'x-amz-date': amzdate, 'x-amz-content-sha256': payload_hash})
        names = sorted(headers)
        canon_headers = ''.join(f'{k}:{headers[k]}\n' for k in names)
        signed = ';'.join(names)
        creq = '\n'.join([method, urllib.parse.quote(path), '', canon_headers, signed, payload_hash])
        scope = f'{date}/auto/s3/aws4_request'
        sts = '\n'.join(['AWS4-HMAC-SHA256', amzdate, scope, hashlib.sha256(creq.encode()).hexdigest()])
        k = ('AWS4' + self.sk).encode()
        for part in (date, 'auto', 's3', 'aws4_request'):
            k = hmac.new(k, part.encode(), hashlib.sha256).digest()
        sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
        headers['Authorization'] = f'AWS4-HMAC-SHA256 Credential={self.ak}/{scope}, SignedHeaders={signed}, Signature={sig}'
        return headers

    def put(self, key, body, ctype, cache):
        if isinstance(body, str):
            body = body.encode('utf-8')
        if DRY:
            p = os.path.join(STATE_DIR, 'dry', key)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'wb').write(body)
            return True
        path = f'/{self.bucket}/{key}'
        h = self._sign('PUT', path, hashlib.sha256(body).hexdigest(), {'content-type': ctype, 'cache-control': cache})
        for attempt in range(3):
            try:
                with self.lock:
                    if self.conn is None:
                        self.conn = http.client.HTTPSConnection(self.ep.netloc, timeout=20)
                    self.conn.request('PUT', urllib.parse.quote(path), body=body, headers=h)
                    r = self.conn.getresponse(); r.read()
                if r.status == 200:
                    return True
                log('R2 PUT', key, 'HTTP', r.status)
            except Exception as e:
                log('R2 PUT', key, 'error', type(e).__name__)
                with self.lock:
                    try: self.conn.close()
                    except Exception: pass
                    self.conn = None
            time.sleep(1 + attempt)
        return False

    def delete(self, key):
        if DRY:
            return True
        path = f'/{self.bucket}/{key}'
        h = self._sign('DELETE', path, hashlib.sha256(b'').hexdigest(), {})
        try:
            with self.lock:
                if self.conn is None:
                    self.conn = http.client.HTTPSConnection(self.ep.netloc, timeout=20)
                self.conn.request('DELETE', urllib.parse.quote(path), headers=h)
                r = self.conn.getresponse(); r.read()
            return r.status in (200, 204, 404)
        except Exception as e:
            log('R2 DELETE', key, 'error', type(e).__name__)
            with self.lock:
                try: self.conn.close()
                except Exception: pass
                self.conn = None
            return False

    # Like the live segments (src/r2/upload.js): each subtitle file is deleted R2_RETENTION_MS after it went up,
    # with the sweeper below as the backstop for anything a restart left behind.
    def expire_later(self, key):
        with self.lock:
            heapq.heappush(self.expiry, (time.time() + RETENTION_S, key))

    def expiry_loop(self):
        while True:
            due = []
            with self.lock:
                while self.expiry and self.expiry[0][0] <= time.time():
                    due.append(heapq.heappop(self.expiry)[1])
            for key in due:
                self.delete(key)
            time.sleep(5)


# ---------------------------------------------------------------- Groq
class Groq:
    API = 'https://api.groq.com/openai/v1'

    def __init__(self, key):
        self.key = key
        self.lock = threading.Lock()
        self.day = None
        self.seconds_today = 0.0

    def _req(self, path, data, ctype, timeout=60):
        req = urllib.request.Request(self.API + path, data=data, headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': ctype, 'User-Agent': 'kctv-subtitles/1'})
        for attempt in range(3):
            try:
                return json.load(urllib.request.urlopen(req, timeout=timeout))
            except urllib.error.HTTPError as e:
                wait = float(e.headers.get('retry-after') or 0) if e.code == 429 else 0
                log('Groq', path, 'HTTP', e.code)
                if e.code in (400, 401, 403, 413):
                    return None
                time.sleep(min(max(wait, 1.5 * (attempt + 1)), 8))
            except Exception as e:
                log('Groq', path, 'error', type(e).__name__)
                time.sleep(1.5 * (attempt + 1))
        return None

    def allow(self, seconds):
        """Spend guard: refuse once today's audio would pass MAX_HOURS_PER_DAY."""
        with self.lock:
            today = time.strftime('%F')
            if today != self.day:
                self.day, self.seconds_today = today, 0.0
            if self.seconds_today + max(seconds, 10) > MAX_HOURS_PER_DAY * 3600:
                return False
            self.seconds_today += max(seconds, 10)
            return True

    def transcribe(self, flac):
        b = uuid.uuid4().hex
        fields = {'model': SPEECH_MODEL, 'language': 'ko', 'response_format': 'verbose_json', 'temperature': '0'}
        body = b''.join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
        body += f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.flac"\r\nContent-Type: audio/flac\r\n\r\n'.encode() + flac + f'\r\n--{b}--\r\n'.encode()
        r = self._req('/audio/transcriptions', body, 'multipart/form-data; boundary=' + b)
        return (r or {}).get('segments') or []

    def translate(self, lines):
        if not lines:
            return []
        user = '\n'.join(f'{i + 1}. {t}' for i, t in enumerate(lines))
        body = json.dumps({'model': TEXT_MODEL, 'temperature': 0, 'response_format': {'type': 'json_object'},
                           'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]}).encode()
        for _ in range(2):
            r = self._req('/chat/completions', body, 'application/json')
            try:
                out = json.loads(r['choices'][0]['message']['content'])['lines']
                if isinstance(out, list) and len(out) == len(lines):
                    return [dprk_english(str(x).strip()) for x in out]
            except Exception:
                pass
        return None


# DPRK house style for the English, shared by both translators: KCNA / Pyongyang Times usage and DPRK romanization.
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


SPOKEN_ENDING = re.compile(r'(니다|니까|십시오|십시요|세요|ㅂ시다|읍시다|습니까|겠습니다)[\s.?!…"\'」』)]*$')


def is_song(kind, ko, en):
    """Stricter than Gemini's label: songs get no subtitles, so a spoken line must never be taken for one.
    Narration, news and commentary end in formal endings (-습니다/-ㅂ니다); lyrics almost never do. A long line
    is narration too."""
    if not str(kind or '').lower().startswith('song'):
        return False
    return not SPOKEN_ENDING.search(str(ko or '').strip()) and len(en) <= 92


# South Korean spellings that slip into translations, and their DPRK forms
SOUTH_TO_DPRK = [(r"\bBaekdu\b", "Paektu"), (r"\bPyeongyang\b", "Pyongyang"), (r"\bHamheung\b", "Hamhung"),
                 (r"\b[GK]aeseong\b", "Kaesong"), (r"\bCheongjin\b", "Chongjin"), (r"\bDaedong\b", "Taedong"),
                 (r"\bSeodaemun\b", "Sodaemun"), (r"\bJoseon\b", "Korea"), (r"\bChosun\b", "Korea"),
                 (r"\bdeodeok\b", "todok"), (r"\bDeodeok\b", "Todok"), (r"\bKim Il-sung\b", "Kim Il Sung"),
                 (r"\bKim Jong-il\b", "Kim Jong Il"), (r"\bKim Jong-un\b", "Kim Jong Un"), (r"\bKim Jong-suk\b", "Kim Jong Suk"),
                 (r"\bRi Sol-ju\b", "Ri Sol Ju"), (r"\bKim Yo-jong\b", "Kim Yo Jong"),
                 (r"\bKimyok\b", "Kim Hyok"), (r"\bKim Hyeok\b", "Kim Hyok")]
# one romanized Korean syllable (DPRK McCune-Reischauer, with the South's eo/eu too), capitalised
KO_SYL = r"(?:Kk|Tt|Pp|Ss|Jj|Ch|Kh|Th|Ph|[KNTRMPSJHGDBL])?(?:yae|wae|ae|ya|eo|eu|yo|ye|wa|oe|wo|we|wi|yu|ui|a|o|e|u|i)(?:ng|[ktlmpn])?"


def dprk_english(t):
    """Safety net for the house style: the DPRK never calls itself 'North Korea'."""
    t = re.sub(r"\bNorth Korean\b", "Korean", t)
    t = re.sub(r"\b[Tt]he North Korea\b|\bNorth Korea\b", "the DPRK", t)
    # DPRK names never take a hyphen: "Pang Thae-so" -> "Pang Thae So" (only a short syllable after a surname, so
    # ordinary English words like "self-reliance" are left alone)
    t = re.sub(r"\b([A-Z][a-z]{1,5}) ([A-Z][a-z]{0,5})-([a-z]{1,5})\b", lambda m: "%s %s %s" % (m.group(1), m.group(2), m.group(3).capitalize()), t)
    for south, dprk in SOUTH_TO_DPRK:
        t = re.sub(south, dprk, t)
    # a time with a one-digit minute (10:6 -> 10:06): only clear times, never a score or ratio like 3:4
    t = re.sub(r"\b(1\d|2[0-3]):(\d)(?!\d)", r"\1:0\2", t)
    t = re.sub(r"(\b[Aa]t |\b[Ff]rom |\b[Uu]ntil )(\d):(\d)(?!\d)", r"\1\2:0\3", t)
    t = re.sub(r"\b([Tt])ravelog\b", r"\1ravelogue", t)
    # a given name written with a hyphen ("Chang-go", "Hyon-jun"): DPRK style writes each syllable as a word
    t = re.sub(r"\b(%s)-(%s)\b" % (KO_SYL, KO_SYL.lower()), lambda m: m.group(1) + ' ' + m.group(2).capitalize(), t)
    return t.replace("the the DPRK", "the DPRK")

# ---------------------------------------------------------------- Gemini (first choice)
SYSTEM_PROMPT = (
    "You translate live North Korean television (KCTV) Korean speech into natural, concise English subtitles. "
    "Fix obvious speech-recognition slips from context. Song lyrics: translate them plainly. " + DPRK_STYLE +
    ' Return ONLY a JSON object {"lines": [...]} with exactly one English string per numbered input line, in the same order.')

GEMINI_PROMPT = (
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
    "SUBTITLE STYLE for the English: concise and natural, sentence case, clock times always as 24-hour digits like the DPRK's own schedule cards (18시 49분 = 18:49, 오후 3시 8분 = 15:08, never '49 minutes past 6'), numbers one to ten in words and 11 upwards in "
    "digits (but times, dates, ages, percentages and anything with a unit always in digits: 3 km, 9 o'clock, 5%), normal punctuation as in broadcast subtitles (a full stop at the end of every sentence; a stretch that ends "
    "mid-sentence has no end punctuation or just its natural comma), no filler or hesitation words (예, 애, 음, 네 as filler -> leave out), no speaker labels; each stretch must "
    "fit in a subtitle of at most 74 characters, so split long sentences into separate stretches at natural clause "
    "breaks. Keep it readable: about 15 characters per second of the stretch at most - condense like a professional "
    "subtitler (drop repetition, honorific padding and filler, keep every fact, name and number) rather than translate "
    "word for word, and never leave anything out of substance. Make the cuts at natural clause "
    "breaks, each with its own times. "
    "Keep each stretch to one sentence or clause (at most about 7 seconds). Mark kind as 'song' ONLY when the words "
    "are actually SUNG to a melody (a song, a choir, sung lyrics); everything spoken is 'speech' — including narration, "
    "a presenter, poetry recitation or dialogue with music playing underneath. When unsure, use 'speech'. "
    "SONGS: give one stretch per sung lyric line (usually 3-7 seconds), starting exactly when that line starts to be "
    "sung, never two lines merged into one, and include every line of every verse and refrain. " + DPRK_STYLE)
GEMINI_SCHEMA = {"type": "OBJECT", "properties": {"segments": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "start": {"type": "NUMBER"}, "end": {"type": "NUMBER"}, "kind": {"type": "STRING"}, "lang": {"type": "STRING"}, "ko": {"type": "STRING"}, "en": {"type": "STRING"}},
    "required": ["start", "end", "kind", "ko", "en"]}}}, "required": ["segments"]}


class Gemini:
    def __init__(self, key):
        self.key = key
        self.until = 0.0
        self.errors = []                                           # times of failed requests (for the admin panel)
        self.lock = threading.Lock()
        self.lookups = 0                                           # dictionary lookups made by Gemini
        try: self.usage = json.load(open(USAGE_FILE))
        except Exception: self.usage = {}

    def count(self, u):
        """Tokens Gemini billed, per day (Pyongyang-independent: server date), kept across restarts."""
        day = time.strftime('%F')
        with self.lock:
            d = self.usage.setdefault(day, {'requests': 0, 'in': 0, 'out': 0})
            d['requests'] += 1
            d['in'] += int(u.get('promptTokenCount') or 0)
            d['out'] += int(u.get('candidatesTokenCount') or 0) + int(u.get('thoughtsTokenCount') or 0)
            for k in sorted(self.usage)[:-40]: self.usage.pop(k)
            try:
                tmp = USAGE_FILE + '.tmp'; json.dump(self.usage, open(tmp, 'w')); os.replace(tmp, USAGE_FILE)
            except Exception: pass

    def today(self):
        d = dict(self.usage.get(time.strftime('%F'), {'requests': 0, 'in': 0, 'out': 0}))
        d['usd'] = round(d['in'] * PRICE_IN / 1e6 + d['out'] * PRICE_OUT / 1e6, 3)
        return d

    def check_terms(self, lines, deadline, intro=None, request_s=25):
        """Second, text-only step: the same model reviews the batch's English with a search_dictionary tool over
        the Samhung dictionaries, looks up whatever it is unsure of (loanwords, technical terms, rare words) and
        returns the corrected lines. Text only, so the audio is not sent again. Any failure keeps the first answer."""
        if not lines or not os.path.exists(DICT_DB):
            return
        items = []
        for i, l in enumerate(lines):
            item = {'n': i + 1, 'korean': l[3], 'english': l[2]}
            hints = [loan_hint(w) for w in loanwords_in(l[3])] + ['%s = %s (place name)' % (k, e) for k, e in places_in(l[3])]
            hints += ['%s = %s (person; DPRK romanization)' % (k, e) for k, e in names_in(l[3])]
            if hints:
                item['dictionary'] = hints                 # found already: saves Gemini a lookup round
            items.append(item)
        prompt = ((intro or "You check English subtitles translated from DPRK television. For each item you have the Korean as heard "
                   "and the current English.") + " Use the search_dictionary tool (the DPRK's own Samhung Korean-English "
                  "dictionary) to look up any Korean word whose translation you are not sure of: DPRK loanwords "
                  "(세멘트 = cement, 뜨락또르 = tractor), technical and scientific terms, rare words. Look up words, not "
                  "whole sentences; you may call the tool several times at once. Do not look up common words or "
                  "personal names. PLACE NAMES: where an item's Korean names a place given in its 'dictionary' hints "
                  "or found with the tool, write it exactly in that DPRK spelling (Mt. Paektu, Sinhung County, Hamhung, "
                  "never South Korean forms like Baekdu or Sinheung) - but only if the Korean really is that place and "
                  "not an ordinary word with the same letters. PERSONAL NAMES given in an item's own hints (DPRK romanization) "
                  "are written exactly that way in that item only - never put a name into an item whose Korean does not "
                  "contain it - and only if the Korean really is that person's name and "
                  "not an ordinary word with the same letters. Then correct the English where it is wrong, romanized or "
                  "too vague, and change "
                  "nothing else: keep the meaning, length and subtitle style (concise, normal punctuation with a full stop at the end of a sentence). " + DPRK_STYLE +
                  " Finally answer with ONLY a JSON object {\"lines\": [...]} holding one English string per item, in order.\n"
                  + json.dumps(items, ensure_ascii=False))
        tools = [{"functionDeclarations": [{"name": "search_dictionary",
                  "description": "Look up a Korean word (DPRK spelling) in the DPRK Samhung Korean-English dictionary. Returns matching headwords with their English meanings and, for loanwords, the foreign original.",
                  "parameters": {"type": "OBJECT", "properties": {"word": {"type": "STRING", "description": "one Korean word, without particles if possible"}}, "required": ["word"]}}]}]
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        url = 'https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent' % GEMINI_MODEL
        try:
            for round_ in range(8):
                left = deadline - time.time()
                if left < 6:
                    return
                body = json.dumps({"contents": contents, "tools": tools, "generationConfig": {"temperature": 0}}).encode()
                req = urllib.request.Request(url, data=body, headers={'x-goog-api-key': self.key, 'Content-Type': 'application/json'})
                try:
                    r = json.load(urllib.request.urlopen(req, timeout=max(5, min(request_s, left - 2))))
                except urllib.error.HTTPError as e:
                    if e.code in (429, 500, 503):                 # busy: try this round again while there is time
                        time.sleep(2); continue
                    raise
                self.count(r.get('usageMetadata') or {})
                content = r['candidates'][0]['content']
                calls = [p['functionCall'] for p in content.get('parts', []) if 'functionCall' in p]
                if calls:
                    contents.append(content)
                    contents.append({"role": "user", "parts": [{"functionResponse": {"name": c['name'], "response": search_dictionary((c.get('args') or {}).get('word')), **({'id': c['id']} if c.get('id') else {})}} for c in calls]})
                    self.lookups += len(calls)
                    continue
                text = ''.join(p.get('text', '') for p in content.get('parts', []))
                m = re.search(r'\{.*\}', text, re.S)
                fixed = json.loads(m.group(0))['lines'] if m else None
                if not fixed or len(fixed) != len(lines):
                    return
                for l, en in zip(lines, fixed):
                    en = str(en or '').strip()
                    if en and en != l[2] and len(en) < 2 * len(l[2]) + 20 and small_fix(l[2], en, l[3]):
                        log('term check:', l[2], '->', en)
                        l[2] = en
                return
        except Exception as e:
            log('term check failed', type(e).__name__)

    def fix_loanwords(self, lines, deadline=None):
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
                  "nothing else and keep the subtitle style (concise, normal punctuation with a full stop at the end of a sentence). " + DPRK_STYLE +
                  ' Return {"lines": [...]} with one English string per item, in order.\n' + json.dumps(items, ensure_ascii=False))
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                                "responseSchema": {"type": "OBJECT", "properties": {"lines": {"type": "ARRAY", "items": {"type": "STRING"}}},
                                                                   "required": ["lines"]}}}).encode()
        req = urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent' % GEMINI_MODEL, data=body,
                                     headers={'x-goog-api-key': self.key, 'Content-Type': 'application/json'})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=max(3, min(15, (deadline or time.time() + 15) - time.time()))))
            self.count(r.get('usageMetadata') or {})
            fixed = json.loads(r['candidates'][0]['content']['parts'][0]['text'])['lines']
            if len(fixed) != len(todo):
                return
            for (i, hints), en in zip(todo, fixed):
                en = str(en or '').strip()
                if en and en != lines[i][2]:
                    log('loanword fix:', lines[i][2], '->', en)
                    lines[i][2] = en
        except Exception as e:
            log('loanword fix failed', type(e).__name__)

    def subtitles(self, flac, seconds, deadline=None, expect_speech=False):
        """[(start, end, english)] for the clip, or None if Gemini could not answer (caller falls back)."""
        if not self.key or time.time() < self.until:
            return None
        body = json.dumps({"contents": [{"parts": [{"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(flac).decode()}},
                                                   {"text": GEMINI_PROMPT}]}],
                           "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": GEMINI_SCHEMA}}).encode()
        # Each try is short and the whole batch has a deadline (its subtitles must be up before viewers reach it):
        # one slow answer used to wait 96 s and retry 5 times, leaving the channel minutes behind.
        deadline = deadline or time.time() + 90
        attempt = -1
        while time.time() < deadline - 3:
            attempt += 1
            model = GEMINI_MODEL
            req = urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent' % model, data=body,
                                         headers={'x-goog-api-key': self.key, 'Content-Type': 'application/json'})
            try:
                r = json.load(urllib.request.urlopen(req, timeout=max(5, min(REQUEST_TIMEOUT, deadline - time.time()))))
                self.count(r.get('usageMetadata') or {})
                segs = json.loads(r['candidates'][0]['content']['parts'][0]['text'])['segments']
                lines = []
                for g in segs:
                    en, ko = str(g.get('en') or '').strip(), str(g.get('ko') or '').strip()
                    st, et = float(g.get('start', 0)), float(g.get('end', 0))
                    if not en or not ko or not_speech(ko, g.get('lang')):
                        continue
                    st = min(max(0.0, st), seconds); et = min(max(et, st + 1.2), seconds + 2)
                    lines.append([st, et, en, ko, g.get('kind')])
                # Gemini sometimes answers with nothing for audio full of speech (3-4% of chunks on 2026-10-16; asked
                # again, most came back transcribed): ask again, up to twice, when the sound says someone is there
                if not lines and expect_speech and attempt < 2 and time.time() < deadline - 15:
                    continue
                # the dictionary check only where it can matter, and never long: batches come every 30 s, and a
                # slow check made the channel fall behind and drop batches when Google was slow
                if time.time() < deadline - 20 and any(loanwords_in(l[3]) or odd_words(l[2], any_case=True)
                                                       or any(place_missing(e, l[2]) for k, e in places_in(l[3]))
                                                       or any(e.split(' ', 1)[-1].lower() not in l[2].lower() for k, e in names_in(l[3])) for l in lines):
                    self.check_terms(lines, min(deadline - 5, time.time() + TERM_CHECK_S))
                out = []
                for st, et, en, ko, kind in lines:
                    en = dprk_english(en)
                    note_unknown_words(en, ko)
                    out.append((st, et, ('♪ ' + en) if is_song(kind, ko, en) else en))
                return out
            except urllib.error.HTTPError as e:
                log('Gemini HTTP', e.code, model)
                self.errors.append(time.time())
                if e.code in (400, 401, 403):
                    return None
                time.sleep(min(1.5 * (attempt + 1), 4))               # busy: wait a little, same model
            except Exception as e:
                log('Gemini error', type(e).__name__, model)
                self.errors.append(time.time())
                time.sleep(1)
        return None


# ---------------------------------------------------------------- recordings: the extra passes used for whole days
def gemini_json(key, parts, schema=None, timeout=120, tries=8):
    """One Gemini request returning JSON (retries while Google is busy). None if it never answered."""
    cfg = {"temperature": 0, "responseMimeType": "application/json"}
    if schema:
        cfg["responseSchema"] = schema
    body = json.dumps({"contents": [{"parts": parts}], "generationConfig": cfg}).encode()
    url = 'https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent' % GEMINI_MODEL
    for attempt in range(tries):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(url, data=body, headers={'x-goog-api-key': key, 'Content-Type': 'application/json'}), timeout=timeout))
            return json.loads(r['candidates'][0]['content']['parts'][0]['text'])
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 402, 403):
                return None
            time.sleep(3 + 2 * attempt)
        except Exception:
            time.sleep(3)
    return None


def speech_or_music(key, flac):
    """(speech, singing) for a stretch of audio: a chunk that came back empty but is full of sound is usually music,
    sometimes speech Gemini skipped (2026-10-16: 144 music, 25 speech among such chunks)."""
    v = gemini_json(key, [{"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(flac).decode()}},
                          {"text": "Is anyone speaking in this audio? Is anyone singing words?"}],
                    {"type": "OBJECT", "properties": {"speech": {"type": "BOOLEAN"}, "singing": {"type": "BOOLEAN"}},
                     "required": ["speech", "singing"]})
    return (bool(v and v.get('speech')), bool(v and v.get('singing')))


def recheck_empty(key, x, seconds, gemini=None):
    """An empty answer for audio full of voice-band sound: ask whether anyone speaks or sings and, if so, transcribe
    again (up to 3 times). Returns new lines (snapped to the voice) or [] for music."""
    if voiced_seconds(x, 0, seconds) <= 0.6 * seconds:
        return []
    flac = to_flac(x)
    if not any(speech_or_music(key, flac)):
        return []
    g = gemini or Gemini(key)
    for attempt in range(3):
        out = g.subtitles(flac, float(seconds), time.time() + 200, True)
        if out:
            return snap_to_speech(out, x)
    return []


def condense_cues(key, cues, cps=20.0):
    """Broadcast practice for subtitles too dense to read: shorten the text to fit its time (at most `cps` characters
    a second), keeping the meaning, every name and every number. Text-only requests in groups of 40; a shortened
    line that loses a name or number, or is still too long, is refused. cues: [(start, end, text)] as make_cues gives."""
    cues = list(cues)
    todo = []
    for i, (st, et, t) in enumerate(cues):
        plain = t.replace('\n', ' ')
        if len(plain) / max(0.1, et - st) > cps:
            todo.append((i, plain.startswith('♪'), plain.strip('♪ ').strip(), max(12, int(cps * (et - st)))))
    def ok(old, new, mx):
        if not new or len(new) > mx + 3:
            return False
        return all(n in new for n in re.findall(r'\d+', old)) and all(n in new for n in re.findall(r'\b[A-Z][a-z]+\b', old)[1:])
    schema = {"type": "OBJECT", "properties": {"lines": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["lines"]}
    done = 0
    for k in range(0, len(todo), 40):
        grp = todo[k:k + 40]
        prompt = ("Shorten each English TV subtitle to at most the given number of characters, as a professional broadcast "
                  "subtitler would: keep the meaning, every name and every number; drop repetition and padding; normal "
                  "punctuation with a full stop at the end of a sentence (keep a missing end mark missing: that line "
                  "continues). Return {\"lines\": [...]} in the same order.\n" +
                  json.dumps([{'max': mx, 'text': txt} for i, song, txt, mx in grp], ensure_ascii=False))
        v = gemini_json(key, [{"text": prompt}], schema)
        out = (v or {}).get('lines') or []
        if len(out) != len(grp):
            continue
        for (i, song, txt, mx), new in zip(grp, out):
            new = dprk_english(str(new).strip())
            if ok(txt, new, mx):
                rows = split_two(new) or [new]
                cues[i] = (cues[i][0], cues[i][1], song_marks(rows) if song else '\n'.join(rows))
                done += 1
    log('shortened %d of %d dense subtitles' % (done, len(todo)))
    return cues


# ---------------------------------------------------------------- audio helpers
def segment_info(path):
    """First (lowest) video and audio PTS, and the audio as 16 kHz mono float32."""
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'packet=codec_type,pts', '-of', 'csv=p=0', path], capture_output=True, text=True, timeout=30)
    vp = ap = None
    for line in p.stdout.splitlines():
        f = line.split(',')
        if len(f) < 2 or not f[1].lstrip('-').isdigit():
            continue
        v = int(f[1])
        if f[0] == 'video' and (vp is None or v < vp): vp = v
        if f[0] == 'audio' and (ap is None or v < ap): ap = v
    a = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-map', '0:a:0', '-ac', '1', '-ar', '16000', '-f', 's16le', '-'], capture_output=True, timeout=30).stdout
    audio = np.frombuffer(a, dtype=np.int16).astype(np.float32) / 32768.0
    return vp, ap, audio


def sound_kind(x):
    """'tone' (an off-air test signal), 'silence', or 'sound'.

    KCTV sends two different test signals before sign-on: a 1 kHz tone (all energy at one frequency) and, later,
    a louder two-tone signal (500 Hz + another, 50 % each) — the second one fooled a one-frequency test and cost
    26 minutes. A test signal is one or a few pure tones at a perfectly constant level; music and speech have their
    energy spread out and their level moving all the time.
    """
    if len(x) < 1600:
        return 'silence'
    rms = 20 * np.log10(np.sqrt(float((x ** 2).mean())) + 1e-9)
    if rms < SILENCE_DB:
        return 'silence'
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 16000)
    total, got, Y = float(X.sum()) + 1e-12, 0.0, X.copy()
    for _ in range(3):                                       # energy in the three strongest narrow peaks
        k = int(Y.argmax()); band = np.abs(f - f[k]) < 25
        got += float(Y[band].sum()); Y[band] = 0
    q = len(x) // 4
    levels = [20 * np.log10(np.sqrt(float((x[i * q:(i + 1) * q] ** 2).mean())) + 1e-9) for i in range(4)]
    steady = max(levels) - min(levels) < 0.5
    return 'tone' if got / total > TONE_SHARE and steady else 'sound'


def to_flac(x):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    return subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 's16le', '-ar', '16000', '-ac', '1', '-i', '-', '-c:a', 'flac', '-f', 'flac', '-'],
                          input=pcm, capture_output=True, timeout=60).stdout


try:
    DICTIONARY = set(w.strip().lower() for w in open('/usr/share/dict/words'))
except Exception:
    DICTIONARY = set()
UNKNOWN_LOG = os.path.join(os.environ.get('SUBS_STATE', '/DATA/subs'), 'unknown-words.log')


DICT_DB = os.environ.get('SUBS_DICT_DB', '/DATA/subs/dict/search.db')
LOANWORD_FILE = os.environ.get('SUBS_LOANWORDS', '/DATA/subs/dict/loanwords.json')
LANG_NAMES = {'영': 'English', '로': 'Russian', '도': 'German', '라': 'Latin', '프': 'French', '이': 'Italian', '그': 'Greek',
              '일': 'Japanese', '네': 'Dutch', '에': 'Spanish', '헤': 'Hebrew', '중': 'Chinese', '싼': 'Sanskrit', '뽀': 'Portuguese'}


def load_loanwords():
    """DPRK loanwords from the DPRK's own dictionaries (Samhung: the Korean dictionary marks each loanword with its origin,
    e.g. 세멘트 [cement《영》], 뜨락또르 [трактор《로》]; the Korean-English one gives the meaning). Built by
    /DATA/subs/dict/build_loanwords.py into {word: {en, from, lang}}."""
    table = {}
    try:                                            # the one database (search.db) holds them; the JSON is a fallback
        import sqlite3
        db = sqlite3.connect('file:%s?mode=ro' % DICT_DB, uri=True)
        table = {w: {'en': en, 'from': o, 'lang': l} for w, en, o, l in db.execute('select word, en, origin, lang from loanwords')}
        db.close()
    except Exception:
        try:
            table = json.load(open(LOANWORD_FILE))
        except Exception as e:
            log('no loanword table', type(e).__name__)
            return {}, []
    return table, sorted(table, key=len, reverse=True)


LOANWORDS, LOANWORD_KEYS = load_loanwords()


PARTICLES = ('에서는', '으로는', '에서', '으로', '에게', '께서', '부터', '까지', '처럼', '들을', '들이', '들의', '들은',
             '을', '를', '이', '가', '은', '는', '에', '의', '로', '과', '와', '도', '만', '들')


def search_dictionary(word):
    """The Samhung dictionaries (DPRK Korean-English KEEK and Korean KK, decoded by digitalnk.com/dics):
    entries for a Korean word, also without its particle and as a prefix, for Gemini's search_dictionary tool."""
    word = re.sub(r'[^가-힣]', '', str(word or ''))[:20]
    if not word:
        return {'results': []}
    import sqlite3
    db = sqlite3.connect('file:%s?mode=ro' % DICT_DB, uri=True)
    tries = [word] + [word[:-len(p)] for p in PARTICLES if word.endswith(p) and len(word) - len(p) >= 2]
    out, seen = [], set()
    def add(rows, how):
        for w, d in rows:
            if w in seen or len(out) >= 6:
                continue
            seen.add(w)
            kk = db.execute('select definition from kk where word=?', (w,)).fetchone()
            origin = re.search(r'\[([^\]]+)\]', kk[0].split('\n', 1)[0]).group(1) if kk and '[' in kk[0].split('\n', 1)[0] else ''
            out.append({'word': w, 'match': how, 'english': re.sub(r'\s+', ' ', d.split('\n', 1)[-1])[:240], 'origin': origin})
    for t in tries:
        add(db.execute('select word, definition from ke where word=?', (t,)).fetchall(), 'exact')
    for t in tries:
        if len(out) < 3 and len(t) >= 2:
            add(db.execute('select word, definition from ke where word like ? order by length(word) limit 4', (t + '%',)).fetchall(), 'starts with')
    places = []
    for t in tries:
        for en, typ in db.execute('select english, type from places where key=? order by rank limit 3', (t,)).fetchall():
            if en not in places:
                places.append(en)
    if places:
        out.append({'word': word, 'match': 'place name (GeoNames, DPRK spelling)', 'english': ' / '.join(places[:3])})
    if not out:                                           # no English entry: the DPRK Korean dictionary's own explanation
        for t in tries:
            for w, d in db.execute('select word, definition from kk where word=?', (t,)).fetchall()[:2]:
                out.append({'word': w, 'match': 'exact (Korean dictionary only)', 'english': '', 'korean_definition': re.sub(r'\s+', ' ', d)[:240]})
    db.close()
    return {'results': out}


def load_places():
    """Counties, cities, provinces and mountains (GeoNames, built by /DATA/subs/places/build_places.py): the ones
    worth checking in every line. Two-syllable keys only for cities and counties (fewer clashes with words)."""
    try:
        import sqlite3
        db = sqlite3.connect('file:%s?mode=ro' % DICT_DB, uri=True)
        rows = db.execute("select key, english from places where major=1 and (length(key) >= 3 or type in "
                          "('ADM1','ADM2','PPLC','PPLA')) order by rank").fetchall()
        db.close()
    except Exception as e:
        log('no place names', type(e).__name__)
        return {}, []
    table = {}
    for k, en in rows:
        table.setdefault(k, en)
    return table, sorted(table, key=len, reverse=True)


PLACES, PLACE_KEYS = {}, []


def places_in(original):
    """Known places named in a Korean line, longest first, not overlapping: [(korean, english)]."""
    found, taken = [], [False] * len(original)
    for k in PLACE_KEYS:
        i = original.find(k)
        if i >= 0 and not any(taken[i:i + len(k)]):
            for j in range(i, i + len(k)): taken[j] = True
            found.append((k, PLACES[k]))
    return found


def place_missing(en_place, english):
    base = re.sub(r'^Mt\. | (County|District|Province)$', '', en_place).split('-')[0].lower()
    return len(base) >= 3 and base not in english.lower()


# ---------------------------------------------------------------- Korean personal names, DPRK romanization
SURNAMES = {'김': 'Kim', '리': 'Ri', '이': 'Ri', '박': 'Pak', '최': 'Choe', '정': 'Jong', '조': 'Jo', '한': 'Han', '윤': 'Yun',
            '장': 'Jang', '오': 'O', '강': 'Kang', '허': 'Ho', '문': 'Mun', '신': 'Sin', '류': 'Ryu', '유': 'Ryu', '홍': 'Hong',
            '전': 'Jon', '양': 'Yang', '량': 'Ryang', '안': 'An', '송': 'Song', '황': 'Hwang', '서': 'So', '주': 'Ju', '권': 'Kwon',
            '차': 'Cha', '로': 'Ro', '노': 'Ro', '림': 'Rim', '임': 'Rim', '렴': 'Ryom', '고': 'Ko', '손': 'Son', '백': 'Paek',
            '남': 'Nam', '우': 'U', '공': 'Kong', '방': 'Pang', '변': 'Pyon', '도': 'To', '현': 'Hyon', '계': 'Kye', '곽': 'Kwak',
            '구': 'Ku', '엄': 'Om', '원': 'Won', '천': 'Chon', '하': 'Ha', '태': 'Thae', '표': 'Phyo', '명': 'Myong', '석': 'Sok',
            '인': 'In', '옥': 'Ok', '지': 'Ji', '진': 'Jin', '채': 'Chae', '함': 'Ham', '봉': 'Pong', '길': 'Kil', '마': 'Ma',
            '배': 'Pae', '심': 'Sim', '탁': 'Thak', '편': 'Phyon', '왕': 'Wang', '위': 'Wi', '연': 'Ryon', '려': 'Ryo'}
_INI = ['k', 'kk', 'n', 't', 'tt', 'r', 'm', 'p', 'pp', 's', 'ss', '', 'j', 'jj', 'ch', 'kh', 'th', 'ph', 'h']
_VOW = ['a', 'ae', 'ya', 'yae', 'o', 'e', 'yo', 'ye', 'o', 'wa', 'wae', 'oe', 'yo', 'u', 'wo', 'we', 'wi', 'yu', 'u', 'ui', 'i']
_FIN = ['', 'k', 'k', 'k', 'n', 'n', 'n', 't', 'l', 'k', 'm', 'l', 'l', 'l', 'p', 'l', 'm', 'p', 'p', 't', 't', 'ng', 't', 't',
        'k', 't', 'p', 't']


def romanize_syllable(ch):
    """One Hangul syllable as the DPRK writes names: 정 Jong, 철 Chol, 혁 Hyok, 태 Thae, 순 Sun."""
    c = ord(ch) - 0xAC00
    if not 0 <= c < 11172:
        return ch
    i, v, f = c // 588, (c % 588) // 28, c % 28
    return (_INI[i] + _VOW[v] + _FIN[f]).capitalize()


def romanize_name(name):
    """허정옥 -> Ho Jong Ok; 리일환 -> Ri Il Hwan (surname by the DPRK table, each given-name syllable a word)."""
    name = re.sub(r'[^가-힣]', '', name)
    if len(name) < 2:
        return ''
    sur = SURNAMES.get(name[0], romanize_syllable(name[0]))
    return ' '.join([sur] + [romanize_syllable(c) for c in name[1:]])


ROLE_BEFORE = ('교원', '교수', '박사', '부교수', '학사', '기자', '선생', '로동자', '작업반장', '분조장', '기사장', '기사', '지배인',
               '당비서', '비서', '위원장', '부위원장', '소장', '실장', '과장', '부장', '연구사', '배우', '가수', '인민배우', '공훈배우',
               '선수', '감독', '학생', '동무', '렬사', '영웅', '로력영웅', '공화국영웅', '작가', '시인', '군인', '중대장', '대대장')
ROLE_AFTER = ('동지', '동무', '선생', '교원', '학생', '선수', '씨', '렬사', '녀사', '어머니', '아버지', '할머니', '할아버지')
NAME_RE = re.compile('(?:(%s)\\s*)?([%s][가-힣]{1,2})(?=\\s*(%s)|(?:은|는|이|가|을|를|의|에게|와|과|도|께서)?(?:[\\s\\-–—,.)」』]|$))' % (
    '|'.join(sorted(ROLE_BEFORE, key=len, reverse=True)), ''.join(SURNAMES), '|'.join(ROLE_AFTER)))


def names_in(original):
    """Personal names in a Korean line, recognised by a role next to them (교원 허정옥, 김철 동무): [(korean, english)].
    Leaders' names are left to the house style."""
    out = []
    for m in NAME_RE.finditer(original):
        if not (m.group(1) or m.group(3)):
            continue
        n = m.group(2)
        if n in ('김일성', '김정일', '김정은', '김정숙') or n in out:
            continue
        out.append((n, romanize_name(n)))
    return out



def small_fix(old, new, original):
    """The dictionary check may only correct terms, never rewrite: it once added words nobody said ("their life" ->
    "their life line"). A change is accepted when it removes a non-English word, brings in a hinted name, place or
    loanword meaning, or changes at most two words without adding any."""
    import difflib
    a, b = old.split(), new.split()
    gone = [w for w in odd_words(old, any_case=True) if w.lower() not in new.lower()]
    hinted = [e for k, e in places_in(original) + names_in(original)] + \
             [LOANWORDS[w].get('en') or '' for w in loanwords_in(original)]
    if any(h and h.lower() not in old.lower() and h.lower() in new.lower() for h in hinted):
        return True
    ops = [(op, i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
           difflib.SequenceMatcher(None, [w.lower().strip('.,;:!?') for w in a], [w.lower().strip('.,;:!?') for w in b]).get_opcodes() if op != 'equal']
    changed = sum(max(n, m) for op, n, m in ops)
    removed = sum(n - m for op, n, m in ops if n > m)          # words taken out and not replaced
    if gone:                                       # a romanized word replaced: allow the words around it to change too
        return changed <= 2 + 2 * len(gone) and len(b) <= len(a) + 2 and removed <= len(gone)
    # otherwise only word-for-word swaps: never a word dropped ("At 10:43" -> "10:43"), never words added
    return changed <= 2 and removed == 0 and len(b) <= len(a)


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


def note_unknown_words(en, original):
    """A romanized loanword ("Saeminto" for cement) is not an English word. Log every word the dictionary does not
    know, with the original, so the LOANWORDS glossary in DPRK_STYLE can be extended from real broadcasts."""
    if not DICTIONARY:
        return
    odd = [w for w in re.findall(r"[A-Za-z][a-z]+", en)
           if not any(x in DICTIONARY for x in (w.lower(), w.lower()[:-1], w.lower()[:-2], w.lower()[:-3]))]
    if odd:
        try:
            with open(UNKNOWN_LOG, 'a') as f:
                f.write('%s\t%s\t%s\t%s\n' % (time.strftime('%F %T'), ' '.join(odd), en, original))
        except Exception:
            pass


def not_speech(original, lang):
    """Gemini's original text is junk. Korean lines get the Korean checks; another language (Ryongnamsan's English,
    German, Russian or Chinese programmes) is only dropped when empty or a single repeated character, since a short
    Chinese line can be two characters and an English presenter may well say 'thank you for watching'."""
    if str(lang or 'ko').lower().startswith('ko') or re.search('[\uac00-\ud7a3]', original):
        return is_hallucination({'text': original})
    core = re.sub(r'[\s.,!?~…·-]', '', original)
    return len(set(core)) <= 1


def is_hallucination(seg):
    t = (seg.get('text') or '').strip()
    if not t:
        return True
    low = t.lower()
    if any(h.lower() in low for h in HALLUCINATIONS):
        return True
    if seg.get('compression_ratio', 0) > 2.4:                       # "아 아 아 아 ..." loops
        return True
    core = re.sub(r'[\s.,!?~…·-]', '', t)
    if len(core) <= 2 or len(set(core)) <= 2:
        return True
    if seg.get('no_speech_prob', 0) > 0.6 and seg.get('avg_logprob', 0) < -0.7:
        return True
    return False


LINE = 37                                                    # characters per line, 2 lines per subtitle (BBC broadcast
                                                             # standard; Netflix allows 42)
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
    """Subtitle text as on broadcast TV: normal punctuation (a sentence ends with a full stop; ? and ! as said),
    no ellipses or filler, single spaces. (Until 2026-10-16 the user wanted no end punctuation; now: full broadcast standard.)"""
    t = re.sub(r'\s+', ' ', str(t)).strip()
    t = t.replace('…', ' ').replace('...', ' ')
    t = re.sub(r'^(Yes|Uh|Um|Well|Ah|Oh),\s+', '', t)                     # filler the translator kept
    t = FILLER.sub(' ', t)                                                # hesitation sounds anywhere ("ern", "erm", "uh")
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.I)                 # stumbles: "this this", "from from"
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'[\s,;:\-–—]+$', '', t)                                  # no dangling comma/dash at the very end
    if t and t[-1] not in '.?!"\'’”)':
        t += '.'                                                          # a finished line is a finished sentence
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
    if best and any(' ' not in x for x in best[1]) and len(text) <= LINE + 4:
        return [text]                                   # a line a little long beats a word on its own
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


LINGER_S = 0.4                                      # a subtitle stays at most this long after its words end
LATE_S = 0.1                                  # most a subtitle may start after its voice (see polish)
LEAD_S = 0.0                                  # a subtitle never shows before its voice (user: 'start early')


def make_cues(lines, onsets=None):
    """Lines (start, end, english) -> finished subtitles (start, end, 'line1\nline2') following subtitle rules:
    songs dropped and overlaps tidied (tidy_lines), clean text, at most 2 balanced lines of 42, long lines split
    at clause breaks with time shared by length, at least 1 s / ~17 characters per second on screen (stretching
    into free time), at most 7 s, and an 80 ms gap so a change of subtitle is visible."""
    cues = []
    for st, et, en in tidy_lines(lines):
        song = en.startswith('♪')
        en = en.lstrip('♪ ').strip()
        # capital only at the start of the sentence; a continuation keeps its lower case
        # a sentence split over several subtitles: the middle parts keep a natural comma, no dash or semicolon
        parts = [re.sub(r'[\s;:\-–—]+$', '', p).strip() for p in split_cues(clean_text(en) if not song else clean_text(en).rstrip('.'))]
        parts = [p for p in parts if re.search(r'[A-Za-z0-9]', p)]
        if not parts:
            continue
        tot = sum(len(p) for p in parts) or 1
        cut, t = [st], st                                    # where each part starts: by its share of the characters,
        for p in parts[:-1]:                                 # then moved to the speaker's pause nearby (onsets: times
            t += (et - st) * len(p) / tot                    # the voice resumes, same clock as the lines)
            if onsets is not None and len(onsets) and not song:
                c = onsets[(onsets >= max(t - 0.8, cut[-1] + 0.8)) & (onsets <= min(t + 0.8, et - 0.6))]
                if len(c):
                    t = float(c[np.argmin(np.abs(c - t))])
            cut.append(t)
        cut.append(et)
        for k, p in enumerate(parts):                        # [4]: where the words end; [5]: starts on the voice
            cues.append([cut[k], cut[k + 1], p, song, cut[k + 1], k == 0])
    cues.sort(key=lambda c: c[0])
    for i, c in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else c[0] + 7.0
        want = max(1.0, len(c[2]) / 17.0)
        c[1] = max(c[1], min(c[0] + want, nxt - 0.08))                     # long enough to read, if there is room
        c[1] = min(c[1], c[0] + 7.0, nxt - 0.08 if i + 1 < len(cues) else c[1])
        dense = len(c[2]) / max(0.1, c[4] - c[0]) > 20                      # a lot to read: may stay a little longer
        c[1] = min(c[1], max(c[4] + (2 * LINGER_S if dense else LINGER_S), c[0] + 0.83))  # never long after the voice
        c[1] = max(c[1], c[0] + 0.83)                                       # broadcast minimum: 5/6 of a second
    cues = polish(cues)
    return [(c[0], c[1], song_marks(split_two(c[2]) or [c[2]]) if c[3] else '\n'.join(split_two(c[2]) or [c[2]])) for c in cues]


def polish(cues):
    """Last pass: the same line twice in a row becomes one subtitle; a subtitle too short to read takes a moment from
    the next one when that one can spare it; nothing overlaps."""
    out = []
    for c in cues:
        if out and c[2].lower() == out[-1][2].lower() and c[0] - out[-1][1] < 1.5:
            out[-1][1] = min(max(out[-1][1], c[1]), out[-1][0] + 7.0); continue
        out.append(list(c))
    # A subtitle that begins a line was snapped to where the voice starts: it may start at most LATE_S later to make
    # room for the one before (a reader notices more than that; 2026-10-16 schedule read 0.6-1 s late). The middle
    # parts of a long line were placed by length, so they may move further.
    late = lambda c: LATE_S if len(c) > 5 and c[5] else 1.0
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if a[1] - a[0] < 0.83 and b[1] - b[0] > 1.6:
            need = min(0.83 - (a[1] - a[0]), (b[1] - b[0]) - 1.2)
            if b[0] - a[1] < need + 0.08:
                b[0] = min(a[1] + need + 0.08, b[0] + late(b)) if b[0] < a[1] + need + 0.08 else b[0]
            a[1] = min(a[1] + need, b[0] - 0.08)
    # too much to read (over 20 characters a second): a small lead-in from free time before (at most 0.3 s, as on
    # broadcast TV), then time the next subtitle can spare (it starts a little later but keeps enough for its own text)
    for i, c in enumerate(out):
        need = len(c[2]) / 17.0 - (c[1] - c[0])
        if need <= 0 or len(c[2]) / max(0.1, c[1] - c[0]) <= 20:
            continue
        free_before = c[0] - (out[i - 1][1] + 0.08 if i else 0)
        lead = max(0.0, min(LEAD_S, free_before, need))
        c[0] -= lead; need -= lead
        if need > 0 and i + 1 < len(out):
            n = out[i + 1]
            spare = (n[1] - n[0]) - max(0.83, len(n[2]) / 17.0)
            take = max(0.0, min(need, spare, late(n)))
            if take > 0:
                n[0] += take
                c[1] = min(c[1] + take, n[0] - 0.08)
    # still under the broadcast minimum (quick dialogue, the next line pinned to its voice): appear a little early,
    # into free time (at most 0.4 s), then take what the subtitle before can spare
    for i, c in enumerate(out):
        if i + 1 < len(out) and out[i + 1][0] - c[1] < 0.08 and out[i + 1][0] - c[0] < 0.91:
            c[1] = out[i + 1][0] - 0.08                     # it runs into the next one: measured as it will be shown
        need = 0.83 - (c[1] - c[0])
        if need <= 0:
            continue
        lead = max(0.0, min(need, 2 * LEAD_S, c[0] - (out[i - 1][1] + 0.08 if i else 0)))
        c[0] -= lead; need -= lead
        if need > 0 and i:
            p = out[i - 1]
            take = max(0.0, min(need, (p[1] - p[0]) - max(0.83, len(p[2]) / 17.0)))
            p[1] -= take; c[0] -= take
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        if b[0] < a[1] + 0.08 and a[1] - a[0] <= 0.9 and b[1] - (a[1] + 0.08) >= 1.2 and a[1] + 0.08 - b[0] <= late(b):
            b[0] = a[1] + 0.08                      # a short line keeps its time; the long next one starts a moment later
        if out[i][0] < out[i - 1][1] + 0.08:
            out[i - 1][1] = max(out[i - 1][0] + 0.5, out[i][0] - 0.08)
            if out[i][0] < out[i - 1][1] + 0.08:                 # still overlapping: the next one starts a moment later
                out[i][0] = out[i - 1][1] + 0.08
                out[i][1] = max(out[i][1], out[i][0] + 0.5)
    return out


def song_marks(rows):
    """Sung lyrics: a music note at the start of the first line and the end of the last (♪ ... ♪)."""
    rows = list(rows)
    rows[0] = '♪ ' + rows[0]
    rows[-1] = rows[-1] + ' ♪'
    return '\n'.join(rows)


def wrap_lines(text, width=LINE):
    return split_two(text) or [text]


def chunk_cues(text, start, end):
    """(kept for callers that place one line at a time) the same rules as make_cues for a single line."""
    return make_cues([(start, end, text)])


def voiced_seconds(x, st, et):
    """How much of [st, et] (seconds into x, 16 kHz) sounds like a voice: 50 ms slices loud enough (above -45 dB)
    with most of their energy in the speech band (300-3400 Hz). Silence, a thud or a click scores ~0."""
    a, b = max(0, int(st * 16000)), min(len(x), int(et * 16000))
    n, hop = 0, 800
    f = np.fft.rfftfreq(hop, 1 / 16000)
    band = (f > 300) & (f < 3400)
    win = np.hanning(hop)
    norm = (win ** 2).sum() * hop / 2
    for i in range(a, b - hop + 1, hop):
        X = np.abs(np.fft.rfft(x[i:i + hop] * win)) ** 2
        # the speech band's own level, whatever the bass underneath does (a voice over loud music still counts)
        if 10 * np.log10(X[band].sum() / norm + 1e-12) > -42:
            n += 1
    return n * hop / 16000


def speech_onsets(x):
    """Times (s) where speech starts in x (16 kHz): phrase-level activity well above the noise floor."""
    hop = 320
    n = len(x) // hop
    if n < 10:
        return np.array([])
    fr = x[:n * hop].reshape(n, hop).astype(np.float32)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    act = db > np.percentile(db, 20) + 10
    act = np.convolve(act.astype(np.int32), np.ones(10, dtype=np.int32), 'same') > 5
    return (np.where(~act[:-1] & act[1:])[0] + 1) * 0.02


def audio_timeline(path):
    """(decoded, timeline) for a recording's first audio track: where each decoded frame's sound starts in the sample
    stream ffmpeg -f s16le gives (frames end to end) and where it plays (its own timestamp, from the track's start).
    They differ: KCTV's audio clock runs ~17 ppm fast against its timestamps, so decoded time gains 0.06 s an hour
    (0.68 s late after 13 h, 2026-10-16), and a damaged frame decodes to nothing (5 that day, 24 ms each). Read from
    the decoded frames themselves (framecrc), so lost frames count too. to_timeline maps."""
    p = subprocess.Popen(['nice', '-n', '19', 'ffmpeg', '-nostdin', '-v', 'quiet', '-i', path, '-map', '0:a:0',
                          '-c:a', 'pcm_s16le', '-f', 'framecrc', '-'], stdout=subprocess.PIPE, text=True)
    sr, pts, n = 48000, [], []
    for line in p.stdout:
        if line.startswith('#'):
            if line.startswith('#tb 0:'):
                sr = int(line.split('/')[-1])                 # time base 1/sample rate: pts and duration in samples
            continue
        f = line.split(',')
        if len(f) >= 4:
            pts.append(int(f[2])); n.append(int(f[3]))
    p.wait()
    if len(pts) < 2:
        return None
    pts, n = np.array(pts, dtype=np.float64), np.array(n, dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(n)[:-1]]) / sr, (pts - pts[0]) / sr


def to_timeline(t, tl):
    """Decoded-audio seconds -> seconds on the recording's own clock (see audio_timeline). tl None: unchanged."""
    if tl is None:
        return t
    return np.interp(t, tl[0], tl[1]) if np.ndim(t) else float(np.interp(t, tl[0], tl[1]))


def pause_ends(x, depth=9.0):
    """Where the voice resumes after a short pause inside continuous speech (between clauses, where the global
    threshold of speech_onsets sees no gap): a dip at least `depth` dB below the speech on both sides within 0.3 s,
    lasting 60 ms or more; the time is where the level climbs back. Times in s into x (16 kHz)."""
    hop = 320
    n = len(x) // hop
    if n < 30:
        return np.array([])
    fr = x[:n * hop].reshape(n, hop).astype(np.float32)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    s = np.convolve(db, np.ones(3) / 3, 'same')
    floor = np.percentile(db, 20)
    out, i = [], 15
    while i < n - 15:
        if s[i] == s[i - 5:i + 6].min():
            d = min(s[i - 15:i].max(), s[i:i + 15].max()) - s[i]
            if d >= depth and s[i:i + 15].max() > floor + 10:
                low = s[i] + d / 2
                if (s[i - 5:i + 6] < low).sum() >= 3:
                    j = i
                    while j < n - 1 and s[j] < low:
                        j += 1
                    out.append(j * 0.02)
                    i = j + 5
                    continue
        i += 1
    return np.array(out)


def voice_starts(x):
    """Every place the voice starts or resumes in x: after silence (speech_onsets) or after a short pause (pause_ends).
    make_cues puts the break between two parts of a long line on one of these."""
    return np.unique(np.round(np.concatenate([speech_onsets(x), pause_ends(x)]), 2))


def align_score(lines, x):
    """How well a chunk's spoken lines sit on the speech: the share whose start is within 0.3 s of a speech onset,
    and how many spoken lines there are (songs are not counted: singing over music has no clean onsets)."""
    spoken = [l for l in lines if not l[2].startswith('♪')]
    on = speech_onsets(x)
    if not spoken or not len(on):
        return 1.0, len(spoken)
    return sum(1 for st, et, en in spoken if float(np.min(np.abs(on - st))) <= 0.3) / len(spoken), len(spoken)


def well_aligned(gemini, flac, x, seconds, deadline, expect, lines, tries=2):
    """Snap a chunk's lines to the voice; if fewer than 60% of its spoken lines then start on speech (Gemini timed the
    chunk badly), ask Gemini again (up to `tries` times, while time allows) and keep the best-aligned answer."""
    best = snap_to_speech(lines, x)
    score, n = align_score(best, x)
    for attempt in range(tries):
        if score >= 0.6 or n < 3 or time.time() > deadline - 20:
            break
        new = gemini.subtitles(flac, seconds, deadline, expect)
        if not new:
            continue
        cand = snap_to_speech(new, x)
        cs, cn = align_score(cand, x)
        if cs > score and cn >= 0.7 * n:                  # better timed, and not by dropping lines
            best, score, n = cand, cs, cn
    return best


def snap_to_speech(lines, x):
    """Gemini's times are rough (~0.5 s). Move each line's start to where speech actually starts nearby (and its end to
    where it stops), from the sound itself: phrase-level activity = frames well above the noise floor, smoothed over
    0.2 s. Lines with no clear edge nearby keep Gemini's time. Times are seconds into x (16 kHz).
    First, a line over (near) silence is dropped: Gemini sometimes invents one there ("This is Korean Central
    Television" over 3 s of silence)."""
    if not lines or len(x) < 16000:
        return lines
    lines = [l for l in lines if voiced_seconds(x, l[0] - 0.3, l[1] + 0.3) >= 0.08]
    if not lines:
        return lines
    hop = 320                                                  # 20 ms
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop).astype(np.float32)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    act = db > np.percentile(db, 20) + 10
    act = np.convolve(act.astype(np.int32), np.ones(10, dtype=np.int32), 'same') > 5
    on = (np.where(~act[:-1] & act[1:])[0] + 1) * 0.02
    off = (np.where(act[:-1] & ~act[1:])[0] + 1) * 0.02
    m = len(x) // 160                                          # 10 ms frames for the fine edges
    fdb = 20 * np.log10(np.sqrt((x[:m * 160].reshape(m, 160).astype(np.float32) ** 2).mean(axis=1)) + 1e-9)
    # Gemini sometimes misplaces a whole chunk by 1-2 s (2026-10-16 12:29-12:31: every line late). Find the one
    # shift (up to +-2.5 s) that puts the most line starts on speech onsets; use it only when clearly better and
    # at least 3 lines agree. Songs are left alone: singing over music has no clean onsets.
    spoken = [l for l in lines if not l[2].startswith('♪')]
    if len(spoken) >= 3 and len(on):
        def score(d):
            return sum(max(0.0, 1 - float(np.min(np.abs(on - (st + d)))) / 0.4) for st, et, en in spoken)
        shifts = np.arange(-2.5, 2.51, 0.1)
        sc = [score(d) for d in shifts]
        k = int(np.argmax(sc))
        d, base = float(shifts[k]), score(0.0)
        if abs(d) >= 0.4 and sc[k] >= base + 1.0 and sc[k] >= 1.3 * max(base, 0.5):
            lines = [(st + d, et + d, en) if not en.startswith('♪') else (st, et, en) for st, et, en in lines]
    out = []
    for st, et, en in lines:
        c = on[(on >= st - 1.0) & (on <= st + 0.6)]
        st2 = float(c[np.argmin(np.abs(c - st))]) if len(c) else st
        # a line Gemini starts on silence: the words come with the first onset after it (2026-10-16: 7% of
        # sentence starts sat 0.6-1 s ahead of the voice, beyond the reach above). Silence here also means a music
        # bed with no voice on it: clearly (8 dB+) quieter than the first 0.4 s after the onset.
        c = on[(on > st) & (on <= min(st + 1.5, et - 0.5))] if not en.startswith('♪') else []
        if len(c):
            i0, i1 = max(0, int(st / 0.02)), int(c[0] / 0.02)
            if voiced_seconds(x, st, st + 0.3) < 0.1 or (i1 - i0 >= 3 and db[i1:i1 + 20].mean() - db[i0:i1].mean() >= 8):
                st2 = float(c[0])
        c = off[(off >= et - 0.6) & (off <= et + 1.0) & (off > st2 + 0.6)]
        et2 = float(c[np.argmin(np.abs(c - et))]) if len(c) else et
        if not en.startswith('♪'):
            st2, et2 = fine_edges(fdb, st2, et2)
        out.append((st2, max(et2, st2 + 0.8), en))
    return out


def fine_edges(fdb, st, et):
    """The phrase-level edges above are 20 ms frames smoothed over 0.2 s: a soft first syllable is missed (the line
    shows ~0.1-0.3 s late) and the end lands on the smoothed tail. Refine both on 10 ms frames: voice = 8 dB above
    the local floor (25th percentile of the 6 s around). A start inside voice walks back to where it began (at most
    0.25 s, so it never reaches into the sentence before); a start on silence walks on to the first voice (0.6 s).
    The end goes back to the last voiced 10 ms (at most 1 s), or on while the voice lasts (0.4 s)."""
    n = len(fdb)
    if not 0 <= int(st * 100) < n or int(et * 100) <= 0:
        return st, et                                    # outside this audio (a line timed past the chunk's end)
    def voiced(k, fl):
        return 0 <= k < n and fdb[k] > fl + 8
    k = int(st * 100)
    fl = np.percentile(fdb[max(0, k - 300):k + 300], 25)
    if voiced(k, fl):
        j = k
        while j > k - 25 and voiced(j - 1, fl): j -= 1
    else:
        j = k
        while j < k + 60 and j < n and not voiced(j, fl): j += 1
        if j >= k + 60 or j >= n: j = k
    st2 = j / 100
    k = min(int(et * 100), n)
    fl = np.percentile(fdb[max(0, k - 300):k + 300], 25)
    if not voiced(k - 1, fl):
        j = k
        while j > max(k - 100, int(st2 * 100) + 50) and not voiced(j - 1, fl): j -= 1
    else:
        j = k
        while j < k + 40 and voiced(j, fl): j += 1
    return st2, max(j / 100, st2 + 0.5)


def tidy_lines(lines):
    """Gemini sometimes gives consecutive lines the same start (or overlapping times). Lines that start together share
    their span one after another, by length; any remaining overlap is trimmed so each line ends where the next begins."""
    # song lines are short; a long "song" line is narration or recitation over music that Gemini mislabelled
    # no subtitles for songs: Gemini times sung lyrics badly (late, or missing). A long "song" line is narration or
    # recitation over music that was mislabelled, so it is kept as ordinary speech.
    # songs are subtitled again (user, 2026-10-16) as ♪ lyrics ♪: the '♪ ' mark stays on the line for make_cues.
    # A line that repeats the one just before it (a refrain heard twice in one answer) is dropped.
    kept = []
    for a, b, t in sorted(lines, key=lambda x: (x[0], x[1])):
        if kept and a - kept[-1][1] < 8 and repeats(t.lstrip('♪ '), [kept[-1][2].lstrip('♪ ')]) and len(content_words(t)) >= 2:
            continue
        kept.append((a, b, t))
    lines = kept
    lines = sorted(lines, key=lambda x: (x[0], x[1]))
    out, i = [], 0
    while i < len(lines):
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
    for k in range(len(out) - 1):
        if out[k][1] > out[k + 1][0]:
            out[k][1] = max(out[k][0] + 0.5, out[k + 1][0])
    return [tuple(x) for x in out]


def vtt_time(s):
    s = max(0.0, s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return '%02d:%02d:%06.3f' % (h, m, sec)


def pts_diff(a, b):
    """a - b on the 33-bit MPEG-TS clock."""
    d = (a - b) % WRAP
    return d - WRAP if d > WRAP // 2 else d


EMPTY_VTT = 'WEBVTT\n'                                            # shared by every segment with nothing to show


def content_words(t):
    return {w for w in re.findall(r"[a-z0-9']+", t.lower()) if len(w) > 3}


def repeats(text, shown):
    """The second pass over a carried segment says again what was just shown ("activities, including cooking
    practice" after "...art activities, and cooking practice"): most of its words are already on screen."""
    words = content_words(text)
    if not words:
        return False
    seen = set().union(*(content_words(t) for t in shown)) if shown else set()
    return len(words & seen) / len(words) >= 0.6


# ---------------------------------------------------------------- one channel
class Channel(threading.Thread):
    def __init__(self, cfg, r2, groq, gemini=None):
        super().__init__(daemon=True, name=cfg['key'])
        self.cfg, self.r2, self.groq, self.gemini = cfg, r2, groq, gemini
        self.seen = {}           # seq -> {name, dur, pdt, disc, discseq, vpts, apts, kind, audio}
        self.vtt = {}            # seq -> finished WebVTT text
        self.pending = []        # seqs with sound, waiting for a batch
        self.published_upto = None
        self.master_done = False
        self.codecs = None
        self.state = 'starting'
        self.last_line = ''
        self.last_line_at = 0
        self.stats = []                                             # (time, ok, lines, seconds) per batch
        self.first_run = True
        # The last segment of each batch is sent again at the start of the next one: a sentence cut by the batch
        # edge was often dropped by both. Lines starting in it are taken from the second pass.
        self.next_carry = None                                     # seq carried into the next batch
        self.inflight = []                                         # batches sent, in order, waiting to be put in place
        import concurrent.futures
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix=cfg['key'] + '-gemini')
        self.carry_cues = []                                       # its cues from the first pass (used if sound ends)
        self.carry_after = None                                    # abs pts: end of the last line already kept
        self.carry_texts = []                                      # the lines shown just before it
        self.last_cue_end = None                                   # abs pts where the last finished line ends
        # CEA-608 captions written into a copy of each segment (cc/index<seq>.ts on R2); the stream server's
        # video.m3u8 uses the copy for the segments listed in cc-<key>.json
        self.cc = cea608.Schedule() if cea608 else None
        self.cc_lock = threading.Lock()
        self.cc_done, self.cc_list = set(), []
        self.cc_r2 = R2(ENV) if r2 is not None else None
        self.cc_file = os.path.join(STATE_DIR, 'cc-%s.json' % cfg['key'])
        self.epg = None
        try:
            self.epg = EpgCapture(cfg['key'])
        except Exception as e:
            log(cfg['key'], 'programme guide capture off:', type(e).__name__)

    # ---- source playlist
    def read_playlist(self):
        try:
            text = open(os.path.join(self.cfg['dir'], 'index.m3u8'), encoding='utf-8').read()
        except OSError:
            return []
        mseq = int((re.search(r'#EXT-X-MEDIA-SEQUENCE:(\d+)', text) or [0, 0])[1])
        dseq = int((re.search(r'#EXT-X-DISCONTINUITY-SEQUENCE:(\d+)', text) or [0, 0])[1])
        segs, dur, pdt, disc, i = [], None, None, False, 0
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('#EXTINF:'):
                dur = float(line[8:].split(',')[0])
            elif line.startswith('#EXT-X-PROGRAM-DATE-TIME:'):
                pdt = line.split(':', 1)[1]
            elif line == '#EXT-X-DISCONTINUITY':
                disc = True
                dseq += 1
            elif line and not line.startswith('#') and dur is not None:
                segs.append({'seq': mseq + i, 'name': line, 'dur': dur, 'pdt': pdt, 'disc': disc, 'discseq': dseq})
                i += 1; dur = pdt = None; disc = False
        return segs

    # ---- analysis of new segments
    def analyse(self, s):
        path = os.path.join(self.cfg['dir'], s['name'])
        try:
            vp, ap, audio = segment_info(path)
        except Exception:
            vp = ap = None; audio = np.zeros(0, np.float32)
        if vp is None:
            s.update(vpts=None, apts=None, kind='missing', audio=None)
            return
        kind = sound_kind(audio)
        s.update(vpts=vp, apts=ap if ap is not None else vp, kind=kind, audio=audio if kind == 'sound' else None)
        if self.codecs is None:
            self.codecs = self.probe_codecs(path)

    def probe_codecs(self, path):
        try:
            p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,codec_name,profile,level,width,height', '-of', 'json', path],
                               capture_output=True, text=True, timeout=20)
            st = json.loads(p.stdout)['streams']
            v = next(x for x in st if x['codec_type'] == 'video')
            prof = {'High': '64', 'Main': '4d', 'Baseline': '42', 'Constrained Baseline': '42'}.get(v.get('profile'), '64')
            return {'codecs': 'avc1.%s00%02x,mp4a.40.2' % (prof, int(v.get('level') or 42)), 'res': '%sx%s' % (v.get('width'), v.get('height'))}
        except Exception:
            return {'codecs': 'avc1.64002a,mp4a.40.2', 'res': '1920x1080'}

    # ---- turning a batch of sound into cues
    # Batches overlap: each is sent to Gemini as soon as its audio is ready, without waiting for the one before
    # (when Google is slow an answer takes 30 s or more, and one-at-a-time made every channel fall behind and drop
    # batches). Answers are put in place strictly in order, because each batch carries its last segment into the next.
    def run_batch(self, seqs, final=False):
        carried = self.next_carry if self.next_carry is not None and self.next_carry in self.seen and self.seen[self.next_carry].get('audio') is not None else None
        self.next_carry = None
        if carried is not None:
            seqs = [carried] + [q for q in seqs if q != carried]
        if not seqs:
            return
        hold = None if final or len(seqs) < 2 else seqs[-1]
        self.next_carry = hold
        entry = {'seqs': seqs, 'carried': carried, 'hold': hold, 'future': None}
        if not (carried is not None and len(seqs) == 1):
            segs = [self.seen[q] for q in seqs]
            audio = np.concatenate([x['audio'] for x in segs])
            offsets, o = [], 0
            for x in segs:
                offsets.append(o); o += len(x['audio'])
            secs = len(audio) / 16000
            deadline = min(x.get('seen_at', time.time()) for x in segs) + BATCH_DEADLINE
            flac = to_flac(audio)
            expect = voiced_seconds(audio, 0, secs) > 0.35 * secs
            def ask():
                # the same passes as a recorded day (dayjob / make_subtitles), each only while there is time left
                t0 = time.time()
                lines = self.gemini.subtitles(flac, secs, deadline, expect) if self.gemini else None
                if lines == [] and expect and self.gemini and time.time() < deadline - 40:
                    lines = recheck_empty(self.gemini.key, audio, secs, self.gemini) or []   # music, or skipped speech
                if lines and self.gemini:
                    lines = well_aligned(self.gemini, flac, audio, secs, deadline, expect, lines)   # badly timed: ask again
                cues = make_cues(snap_to_speech(lines, audio), voice_starts(audio)) if lines else []
                if cues and self.gemini and time.time() < deadline - 20:
                    cues = condense_cues(self.gemini.key, cues)                         # too dense to read: shorten
                return (None if lines is None else cues), time.time() - t0
            entry.update(segs=segs, audio=audio, offsets=offsets, future=self.pool.submit(ask))
        self.inflight.append(entry)

    def drain(self):
        while self.inflight and (self.inflight[0]['future'] is None or self.inflight[0]['future'].done()):
            self.complete(self.inflight.pop(0))

    def complete(self, e):
        seqs, carried, hold = e['seqs'], e['carried'], e['hold']
        if e['future'] is None:
            # sound ended right after a batch: the carried segment keeps what the first pass found
            self.finish(carried, self.carry_cues); self.seen[carried]['audio'] = None
            return
        try:
            lines, took = e['future'].result()
        except Exception as ex:
            log(self.cfg['key'], 'batch error', type(ex).__name__)
            lines, took = None, 0
        if lines is None:
            log(self.cfg['key'], 'batch given up: no answer before its deadline')
        self.stats.append((time.time(), lines is not None, len(lines or []), took))
        segs, audio, offsets = e['segs'], e['audio'], e['offsets']
        cues = {q: [] for q in seqs}
        for st, et, text in (lines or []):                         # finished subtitles from the worker (ask)
            self.last_line, self.last_line_at = text.replace('\n', ' '), time.time()
            self.place_cue(segs, offsets, cues, st, et, text)
        if carried is not None:
            if lines is not None:
                # only real repeats of what the first pass already showed go
                cues[carried] = [c for c in cues[carried] if not repeats(c[2], self.carry_texts)]
            else:
                cues[carried] = self.carry_cues
        self.no_overlap(seqs, cues, hold)
        for q in seqs:
            if q == hold:
                continue
            self.finish(q, cues[q]); self.seen[q]['audio'] = None
        if hold is not None:
            self.carry_cues = cues[hold]
            self.carry_texts = [c[2] for q in seqs[-3:-1] for c in cues[q]]

    def no_overlap(self, seqs, cues, hold):
        """Two subtitles must never be on screen together. Batches meet where the carried segment is sent twice:
        lines of this pass end before the carried segment starts (the next pass covers it), and lines of the second
        pass over a carried segment start only after the last line already shown has ended."""
        gap = int(0.08 * TICK)
        def absolute(q, t):
            return self.seen[q]['vpts'] + int(t * TICK)
        if hold is not None and self.seen[hold].get('vpts') is not None:
            limit = self.seen[hold]['vpts'] - gap
            for q in seqs:
                if q == hold or self.seen[q].get('vpts') is None:
                    continue
                kept = []
                for a, b, t in cues[q]:
                    if pts_diff(absolute(q, b), limit) > 0:
                        b = max(a + 0.5, pts_diff(limit, self.seen[q]['vpts']) / TICK)
                    kept.append((a, b, t))
                cues[q] = kept
        last_end = self.last_cue_end
        for q in seqs:
            if self.seen[q].get('vpts') is None:
                continue
            kept = []
            for a, b, t in sorted(cues[q]):
                if last_end is not None and pts_diff(absolute(q, a), last_end + gap) < 0:
                    a = pts_diff(last_end + gap, self.seen[q]['vpts']) / TICK
                    if b - a < 0.7:
                        continue                                    # nothing left of it worth showing
                kept.append((a, b, t))
                last_end = absolute(q, b)
            cues[q] = kept
        # remember where the last finished line ends (a carried segment's lines are decided next time)
        for q in seqs:
            if q == hold or self.seen[q].get('vpts') is None:
                continue
            for c in cues[q]:
                e = absolute(q, c[1])
                if self.last_cue_end is None or pts_diff(e, self.last_cue_end) > 0:
                    self.last_cue_end = e

    def place_cue(self, segs, offsets, cues, cs, ce, txt):
        """Put one finished subtitle (seconds into the batch audio) into the video segment it starts in."""
        k = max(i for i in range(len(segs)) if offsets[i] / 16000 <= cs + 1e-6)
        abs_start = segs[k]['apts'] + int((cs - offsets[k] / 16000) * TICK)
        host = None
        for s in segs:
            d = pts_diff(abs_start, s['vpts'])
            if 0 <= d < s['dur'] * TICK + TICK // 2:
                host = s; break
        if host is None:
            host = segs[k]
        local = pts_diff(abs_start, host['vpts']) / TICK
        cues[host['seq']].append((local, local + (ce - cs), txt))

    def place(self, segs, offsets, cues, start, end, en):
        """Put one subtitle line (seconds into the batch audio) into the video segment(s) it starts in."""
        for (cs, ce, txt) in chunk_cues(en, start, end):
            k = max(i for i in range(len(segs)) if offsets[i] / 16000 <= cs + 1e-6)
            abs_start = segs[k]['apts'] + int((cs - offsets[k] / 16000) * TICK)
            host = None
            for s in segs:
                d = pts_diff(abs_start, s['vpts'])
                if 0 <= d < s['dur'] * TICK + TICK // 2:
                    host = s; break
            if host is None:
                host = segs[k]
            local = pts_diff(abs_start, host['vpts']) / TICK
            cues[host['seq']].append((local, local + (ce - cs), txt))

    def finish(self, seq, cues):
        s = self.seen[seq]
        if CC_ENABLED and s.get('vpts') is not None:
            with self.cc_lock:
                if s.get('disc'):
                    self.cc.reset()
                base = self.cc.unwrap(s['vpts'])
                for a, b, t in sorted(cues):
                    self.cc.add(base + int(a * TICK), base + int(b * TICK), t)
                self.cc.close_until(base + int(s['dur'] * TICK))
        head = 'WEBVTT\n'
        if s.get('vpts') is not None:
            head += 'X-TIMESTAMP-MAP=MPEGTS:%d,LOCAL:00:00:00.000\n' % (s['vpts'] % WRAP)
        if not cues:
            # nothing said in this segment (off air, music, silence): point at one shared empty file instead of
            # uploading another — most segments of the day, and every upload is an R2 write
            self.vtt[seq] = EMPTY_VTT
            return
        body = ''.join('\n%s --> %s\n%s\n' % (vtt_time(a), vtt_time(b), bold_leaders(t.replace('&', '&amp;').replace('<', '&lt;'))) for a, b, t in sorted(cues))
        self.vtt[seq] = head + body
        key = '%s/subs/en%d.vtt' % (self.cfg['prefix'], seq)
        if self.r2.put(key, self.vtt[seq], 'text/vtt; charset=utf-8', 'public, max-age=86400, immutable'):
            self.r2.expire_later(key)

    # ---- publishing
    def publish(self, current):
        # from our own history, not ffmpeg's 10-entry playlist: the subtitle playlist has to reach back past the
        # held-back picture (src/r2/held.js), further than the live playlist does
        current = [self.seen[q] for q in sorted(self.seen) if 'name' in self.seen[q]]
        ready = [s for s in current if s['seq'] in self.vtt]
        upto = None
        for s in current:                                  # contiguous run of finished segments from the start
            if s['seq'] in self.vtt: upto = s['seq']
            else: break
        if upto is None or upto == self.published_upto:
            return
        win = [s for s in ready if s['seq'] <= upto][-WINDOW:]
        if not win:
            return
        first = win[0]
        def playlist(entry):
            out = ['#EXTM3U', '#EXT-X-VERSION:6', '#EXT-X-TARGETDURATION:%d' % max(6, int(max(s['dur'] for s in win) + 0.999)),
                   '#EXT-X-MEDIA-SEQUENCE:%d' % first['seq'], '#EXT-X-DISCONTINUITY-SEQUENCE:%d' % first['discseq']]
            for s in win:
                if s['disc'] and s is not first:
                    out.append('#EXT-X-DISCONTINUITY')
                out.append('#EXTINF:%.6f,' % s['dur'])
                if s.get('pdt'):
                    out.append('#EXT-X-PROGRAM-DATE-TIME:' + s['pdt'])
                out.append(entry(s))
            return '\n'.join(out) + '\n'
        pfx = self.cfg['prefix']
        # Only the subtitle playlist: index.m3u8 (the master) and video.m3u8 (the held-back picture) are written by the
        # stream server's own uploader (src/r2/held.js), so the picture keeps playing even if this service stops.
        subs = playlist(lambda s: 'empty.vtt' if self.vtt.get(s['seq']) is EMPTY_VTT else 'en%d.vtt' % s['seq'])
        if self.r2.put(pfx + '/subs/en.m3u8', subs, 'application/vnd.apple.mpegurl', 'public, max-age=1, must-revalidate'):
            self.published_upto = upto
        if False:                                                           # master now comes from the uploader
            master = ('#EXTM3U\n#EXT-X-VERSION:6\n#EXT-X-INDEPENDENT-SEGMENTS\n'
                      '#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="English",LANGUAGE="en",DEFAULT=YES,AUTOSELECT=YES,FORCED=NO,URI="subs/en.m3u8"\n'
                      '#EXT-X-STREAM-INF:BANDWIDTH=9000000,AVERAGE-BANDWIDTH=8000000,CODECS="%s",RESOLUTION=%s,FRAME-RATE=50.000,SUBTITLES="subs"\n'
                      'index-sub.m3u8\n') % (self.codecs['codecs'], self.codecs['res'])
            self.master_done = self.r2.put(pfx + '/master.m3u8', master, 'application/vnd.apple.mpegurl', 'public, max-age=60')

    # ---- CEA-608 captions inside the video
    def cc_loop(self):
        while True:
            try:
                self.write_cc()
            except Exception as e:
                log(self.cfg['key'], 'cc error', type(e).__name__, str(e)[:200])
            time.sleep(0.5)

    def write_cc(self):
        """Caption segments in order, once their subtitles and the next segment's are known (a caption starting
        early in the next segment is loaded during this one), or when the deadline comes."""
        for q in sorted(q for q in list(self.vtt) if q not in self.cc_done):
            s = self.seen.get(q)
            if s is None or s.get('vpts') is None:
                self.cc_done.add(q); continue
            if q + 1 not in self.vtt and time.time() - s.get('seen_at', 0) < CC_DEADLINE:
                return
            self.cc_done.add(q)
            try:
                data = open(os.path.join(self.cfg['dir'], s['name']), 'rb').read()
            except OSError:
                continue
            def pairs_for(pts):
                with self.cc_lock:
                    u = [self.cc.unwrap(p) for p in pts]
                    got = self.cc.take(u)
                return {p: got[x] for p, x in zip(pts, u) if x in got}
            new, n = cea608.rewrite(data, pairs_for)
            if not n:
                continue                                        # nothing to show or clear: the original serves
            if self.cc_r2.put('%s/cc/%s' % (self.cfg['prefix'], s['name']), new, 'video/mp2t', 'public, max-age=86400, immutable'):
                self.cc_list = (self.cc_list + [q])[-80:]
                tmp = self.cc_file + '.tmp'
                json.dump({'seqs': self.cc_list, 'updated': int(time.time())}, open(tmp, 'w'))
                os.replace(tmp, self.cc_file)

    # ---- main loop
    def run(self):
        log(self.cfg['key'], 'started')
        if CC_ENABLED:
            threading.Thread(target=self.cc_loop, daemon=True, name=self.cfg['key'] + '-cc').start()
        self.r2.put(self.cfg['prefix'] + '/subs/empty.vtt', EMPTY_VTT, 'text/vtt; charset=utf-8', 'public, max-age=86400')
        while True:
            try:
                self.step()
            except Exception as e:
                log(self.cfg['key'], 'error', type(e).__name__, str(e)[:200])
                time.sleep(2)
            time.sleep(1)

    def step(self):
        current = self.read_playlist()
        if not current:
            return
        if self.first_run:
            # A (re)start subtitles what is already on disk too: viewers are held back 108 s and ffmpeg's playlist
            # holds only the last minute, so all of it is still ahead of them. Skipping it blanked the subtitles for
            # ~2.5 minutes after every restart. Only segments too old to make it in time get empty subtitles.
            for s in current:
                try:
                    age = time.time() - os.path.getmtime(os.path.join(self.cfg['dir'], s['name']))
                except OSError:
                    age = 1e9
                if age > BATCH_DEADLINE - 25:
                    self.seen[s['seq']] = dict(s, vpts=None, apts=None, kind='old', audio=None)
                    self.finish(s['seq'], [])
                else:
                    s['seen_at'] = time.time() - age
            self.first_run = False
        for s in current:
            if s['seq'] not in self.seen:
                d = dict(s, seen_at=s.get('seen_at', time.time()))
                self.analyse(d)
                self.seen[d['seq']] = d
                if self.epg and not DRY:
                    try:
                        self.epg.segment(os.path.join(self.cfg['dir'], d['name']), d['kind'])
                    except Exception as e:
                        log(self.cfg['key'], 'programme guide capture error', type(e).__name__)
                if d['kind'] == 'sound':
                    self.pending.append(d['seq'])
                else:
                    self.flush(final=True)                      # sound ended: send what is waiting first
                    self.finish(d['seq'], [])
        # batch: enough sound, or the oldest waiting segment is getting old
        if self.pending:
            secs = sum(self.seen[q]['dur'] for q in self.pending)
            if secs >= BATCH_S or len(current) and self.pending[0] < current[-1]['seq'] - int(BATCH_S / 6 + 2):
                self.flush()
        on = any(self.seen[s['seq']].get('kind') == 'sound' for s in current[-3:] if s['seq'] in self.seen)
        self.state = 'on air' if on else 'off air'
        self.drain()                                            # answers that have come back, in order
        self.publish(current)
        # forget what has left the window
        low = current[-1]['seq'] - WINDOW - 10
        for q in [q for q in self.seen if q < low]:
            self.seen.pop(q, None); self.vtt.pop(q, None); self.cc_done.discard(q)

    def flush(self, final=False):
        if self.pending or (final and self.next_carry is not None):
            batch, self.pending = self.pending, []
            self.run_batch(batch, final)


# ---------------------------------------------------------------- programme guide capture
EPG_DIR = os.path.join(STATE_DIR, 'epg')
EPG_RING_S = 1800                                           # the last 30 minutes of every channel, as hard links
EPG_STATUS = os.path.join(EPG_DIR, 'status.json')
SIGNON_QUIET_S, SIGNOFF_QUIET_S, MIN_ON_AIR_S = 900, 180, 1800


class EpgCapture:
    """The channels show their programme schedule as on-screen cards: at sign-on 오늘의 순서 (today), and KCTV before
    sign-off 래일의 순서 (tomorrow), Ryongnamsan on Friday 다음주 월요일순서. This keeps each channel's last 30 minutes
    of live segments (hard links: no copying) and, 15 minutes after sign-on and right after sign-off, runs epg.py on
    that stretch in its own low-priority process: it finds the cards, reads them and writes the guide for the date
    the heading names (/DATA/subs/epg/out). On air / off air comes from the sound, as for the subtitles."""

    def __init__(self, key):
        self.key = key
        self.ring = os.path.join(EPG_DIR, 'live', key)
        os.makedirs(self.ring, exist_ok=True)
        self.on_since = self.quiet_since = None
        self.opening_due = None
        self.last_sound = 0
        st = self.status()
        self.signoff_hm = st.get(key, {}).get('signoff', '')

    def status(self):
        try:
            return json.load(open(EPG_STATUS))
        except Exception:
            return {}

    def note(self, **kw):
        st = self.status()
        st.setdefault(self.key, {}).update(kw, at=time.strftime('%F %T'))
        try:
            tmp = EPG_STATUS + '.tmp'; json.dump(st, open(tmp, 'w'), ensure_ascii=False, indent=1); os.replace(tmp, EPG_STATUS)
        except Exception:
            pass

    def segment(self, path, kind):
        now = time.time()
        try:
            os.link(path, os.path.join(self.ring, '%d-%s' % (int(now), os.path.basename(path))))
        except OSError:
            pass
        for f in os.listdir(self.ring):
            try:
                if now - int(f.split('-', 1)[0]) > EPG_RING_S:
                    os.unlink(os.path.join(self.ring, f))
            except (ValueError, OSError):
                pass
        if kind == 'sound':
            if self.on_since is None and (self.quiet_since is None or now - self.quiet_since >= SIGNON_QUIET_S):
                self.on_since = now
                if self.quiet_since is not None:              # a real sign-on, not the service starting mid-day
                    self.opening_due = now + 900
                    self.note(signon=pyongyang_hm(now))
            self.quiet_since = None
            self.last_sound = now
        else:
            if self.quiet_since is None:
                self.quiet_since = now
            if self.on_since is not None and now - self.quiet_since >= SIGNOFF_QUIET_S:
                if self.last_sound - self.on_since >= MIN_ON_AIR_S:
                    self.signoff_hm = pyongyang_hm(self.last_sound)
                    self.note(signoff=self.signoff_hm)
                    self.run('closing', self.last_sound)
                self.on_since = None
        if self.opening_due and now >= self.opening_due:
            self.opening_due = None
            self.run('opening', now)

    def run(self, kind, until):
        files = sorted(f for f in os.listdir(self.ring) if int(f.split('-', 1)[0]) <= until + 6)
        if len(files) < 20:
            return
        out = os.path.join(EPG_DIR, 'capture-%s-%s-%s.ts' % (self.key, kind, time.strftime('%Y%m%d-%H%M')))
        with open(out, 'wb') as fo:
            for f in files:
                with open(os.path.join(self.ring, f), 'rb') as fi:
                    fo.write(fi.read())
        day = pyongyang_day(until)
        log_path = out.replace('.ts', '.log')
        cmd = ['nice', '-n', '15', sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'epg.py'),
               'auto', self.key, day, out] + ([self.signoff_hm] if self.signoff_hm else [])
        log(self.key, 'programme guide: reading the', kind, 'schedule from', len(files), 'segments')
        self.note(**{kind: {'started': time.strftime('%F %T'), 'capture': out, 'log': log_path}})
        def go():
            with open(log_path, 'w') as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            self.note(**{kind: {'finished': time.strftime('%F %T'), 'ok': r.returncode == 0, 'capture': out, 'log': log_path}})
            if r.returncode == 0:
                try: os.unlink(out)
                except OSError: pass
        threading.Thread(target=go, daemon=True).start()


def pyongyang_hm(t):
    return time.strftime('%H:%M', time.gmtime(t + 9 * 3600))


def pyongyang_day(t):
    return time.strftime('%Y-%m-%d', time.gmtime(t + 9 * 3600))


# ---------------------------------------------------------------- housekeeping
def sweeper():
    """Backstop, as for the live segments (src/r2/sweep.js): any subtitle file older than the retention, e.g. one
    whose timed delete was lost in a restart. Leftover cc/ copies from the old CEA-608 captions go too."""
    while not DRY:
        age = '%ds' % RETENTION_S
        for ch in CHANNELS:
            subprocess.run(['rclone', 'delete', 'r2:%s/%s/subs' % (ENV.get('R2_BUCKET', 'live'), ch['prefix']), '--use-server-modtime',
                            '--max-depth', '1', '--include', 'en*.vtt', '--min-age', age], capture_output=True, timeout=300)
            subprocess.run(['rclone', 'delete', 'r2:%s/%s/cc' % (ENV.get('R2_BUCKET', 'live'), ch['prefix']), '--use-server-modtime',
                            '--max-depth', '1', '--include', 'index*.ts', '--min-age', age], capture_output=True, timeout=300)
        time.sleep(300)


def status_writer(chans, groq, gemini):
    while True:
        try:
            now = time.time()
            gemini.errors[:] = [t for t in gemini.errors if now - t < 3600]
            ch = {}
            for c in chans:
                c.stats[:] = [x for x in c.stats if now - x[0] < 3600]
                cur = c.read_playlist()
                edge = cur[-1]['seq'] if cur else None
                ch[c.cfg['key']] = {
                    'state': c.state, 'last_line': c.last_line[:200],
                    'last_line_age_s': int(now - c.last_line_at) if c.last_line_at else None,
                    'batches_last_hour': len(c.stats), 'failed_last_hour': sum(1 for x in c.stats if not x[1]),
                    'lines_last_hour': sum(x[2] for x in c.stats),
                    'request_s_avg': round(sum(x[3] for x in c.stats) / len(c.stats), 1) if c.stats else None,
                    # how far the published subtitles trail the live edge; players are held back HOLD_SEGMENTS
                    'behind_live_segments': (edge - c.published_upto) if (edge is not None and c.published_upto) else None,
                }
            st = {'at': time.strftime('%F %T'), 'updated': int(now), 'model': GEMINI_MODEL, 'backup_model': GEMINI_BACKUP,
                  'usage_today': gemini.today(), 'price_per_million': {'in': PRICE_IN, 'out': PRICE_OUT},
                  'hold_segments': HOLD_SEGMENTS, 'batch_s': BATCH_S, 'gemini_errors_last_hour': len(gemini.errors), 'channels': ch}
            tmp = os.path.join(STATE_DIR, '.status.json.tmp')
            open(tmp, 'w').write(json.dumps(st, ensure_ascii=False, indent=1))
            os.chmod(tmp, 0o644)
            os.replace(tmp, os.path.join(STATE_DIR, 'status.json'))
        except Exception as e:
            log('status error', type(e).__name__, str(e)[:120])
        time.sleep(10)


if __name__ == '__main__':
    os.makedirs(STATE_DIR, exist_ok=True)
    ENV = load_env(ENV_FILE)
    PLACES, PLACE_KEYS = load_places()
    RETENTION_S = int(ENV.get('R2_RETENTION_MS') or 900000) / 1000   # same as the video segments (15 min), user's choice
    conf = json.load(open(CONFIG))
    key = conf['groq_key']
    only = [a for a in sys.argv[1:] if not a.startswith('--')]
    chans_cfg = [c for c in CHANNELS if not only or c['key'] in only]
    for c in sys.argv:
        if c.startswith('--dir='):                              # testing: read a different folder
            chans_cfg = [dict(chans_cfg[0], dir=c.split('=', 1)[1])]
    r2, groq = R2(ENV), Groq(key)
    gemini = Gemini(conf.get('gemini_key'))
    chans = [Channel(c, r2, groq, gemini) for c in chans_cfg]
    for c in chans:
        c.start()
    threading.Thread(target=r2.expiry_loop, daemon=True).start()
    threading.Thread(target=sweeper, daemon=True).start()
    threading.Thread(target=status_writer, args=(chans, groq, gemini), daemon=True).start()
    log('subtitles running for', ', '.join(c.cfg['key'] for c in chans), '(dry run)' if DRY else '')
    while True:
        time.sleep(3600)
