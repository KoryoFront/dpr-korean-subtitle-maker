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
import base64, datetime, difflib, heapq, hashlib, hmac, http.client, json, os, re, subprocess, sys, threading, time, urllib.parse, urllib.request, uuid
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
VERTEX_MODELS = {'gemini-flash-lite-latest': 'gemini-3.5-flash-lite'}   # Vertex has no "-latest" alias


def gemini_request(key, model, body):
    """One Gemini call, by whichever service the key belongs to: a Vertex AI key (starts "AQ.", billed to the Google
    Cloud project, so Cloud credits apply) goes to Vertex, an AI Studio key to the Gemini API. Same model, same request
    (Vertex wants every message to name its role, so a missing one is set to "user")."""
    if str(key).startswith('AQ.'):
        b = json.loads(body)
        for c in b.get('contents', []):
            c.setdefault('role', 'user')
        url = 'https://aiplatform.googleapis.com/v1/publishers/google/models/%s:generateContent' % VERTEX_MODELS.get(model, model)
        body = json.dumps(b).encode()
    else:
        url = 'https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent' % model
    return urllib.request.Request(url, data=body, headers={'x-goog-api-key': key, 'Content-Type': 'application/json'})


def config_key(conf):
    """The Gemini key to use: the Vertex AI key when there is one (user, 2026-10-17), else the AI Studio key."""
    return conf.get('vertex_key') or conf.get('gemini_key')
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
# Live subtitles off (user, 2026-10-18: "dont do live subs for now just disable it but keep it, just do the auto
# epg"): the service still runs for the programme guide capture and still publishes each channel's English track
# (the master playlist lists it), with every segment empty. LIVE_SUBS=1 in the environment turns them back on.
LIVE_SUBS = os.environ.get('LIVE_SUBS', '0') == '1'
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


NARRATED_ENDING = re.compile(r'(었다|았다|였다|했다|되었다|하였다|있었다|입니다)[\s.?!…"\'」』)]*$')   # narrative past tense


def is_song(kind, ko, en):
    """Stricter than Gemini's label: songs get no subtitles, so a spoken line must never be taken for one.
    Narration, news and commentary end in formal endings (-습니다/-ㅂ니다); lyrics almost never do. A long line
    is narration too."""
    if not str(kind or '').lower().startswith('song'):
        return False
    ko, en = str(ko or '').strip(), str(en or '')
    if SPOKEN_ENDING.search(ko) or NARRATED_ENDING.search(ko) or len(en) > 92:
        return False
    # narration over music reports facts: a year or a large figure is never sung (2026-10-16 2:45: "visited
    # educators' apartments three times in 2013 alone" came out as a lyric)
    if re.search(r'\d{1,2}:\d\d|\d+\s*시\s*\d*\s*분?|\d+\s*월\s*\d+\s*일', en + ' ' + ko):
        return False                     # a clock time or a date is announced, never sung (the day's schedule over music)
    return not re.search(r'\b(1[89]|20)\d\d\b|\d{3,}|\d+(\.\d+)?\s*(per ?cent|%|tons?|metres|hectares|chongbo)', en)


# South Korean spellings that slip into translations, and their DPRK forms
SOUTH_TO_DPRK = [(r"\bBaekdu\b", "Paektu"), (r"\bPyeongyang\b", "Pyongyang"), (r"\bHamheung\b", "Hamhung"),
                 (r"\b[GK]aeseong\b", "Kaesong"), (r"\bCheongjin\b", "Chongjin"), (r"\bDaedong\b", "Taedong"), (r"\bBusan\b", "Pusan"),
                 (r"\bSeodaemun\b", "Sodaemun"), (r"\bJoseon\b", "Korea"), (r"\bChosun\b", "Korea"),
                 (r"\bdeodeok\b", "todok"), (r"\bDeodeok\b", "Todok"), (r"\bKim Il-sung\b", "Kim Il Sung"),
                 (r"\bKim Jong-il\b", "Kim Jong Il"), (r"\bKim Jong-un\b", "Kim Jong Un"), (r"\bKim Jong-suk\b", "Kim Jong Suk"),
                 (r"\bRi Sol-ju\b", "Ri Sol Ju"), (r"\bKim Yo-jong\b", "Kim Yo Jong"),
                 (r"\bKimyok\b", "Kim Hyok"), (r"\bKim Hyeok\b", "Kim Hyok"),
                 # 타도제국주의동맹 (ㅌ.ㄷ) heard as a name (2026-10-17 20:00 news: "the Paedojio Football Alliance")
                 # 리, the distance (three-thousand-ri land, 130 ri): DPRK English writes ri; capital Li (Chinese
                 # names, Li Qiang) is left alone (user, 2026-10-18)
                 (r"(?<![A-Za-z'])li\b", "ri"),
                 # the distance always with a space before it (user, 2026-10-18): "thousand-ri" -> "thousand ri", "130ri"
                 # -> "130 ri" (only after a number: a village keeps its hyphen, Kuup-ri; Henri stays Henri)
                 (r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|hundred|hundreds|thousand|thousands|million|millions|many|several)-?ri\b", r"\1 ri"),
                 (r"\b(?:Paedojio|Thadoje(?:guk)?(?:ju)?(?:ui)?)(?: Football)? (?:Alliance|Union|League)\b", "Down-with-Imperialism Union")]
# one romanized Korean syllable (DPRK McCune-Reischauer, with the South's eo/eu too), capitalised
KO_SYL = r"(?:Kk|Tt|Pp|Ss|Jj|Ch|Kh|Th|Ph|[KNTRMPSJHGDBL])?(?:yae|wae|ae|ya|eo|eu|yo|ye|wa|oe|wo|we|wi|yu|ui|a|o|e|u|i)(?:ng|[ktlmpn])?"


def dprk_english(t):
    """Safety net for the house style: the DPRK never calls itself 'North Korea'."""
    # invisible control characters never reach a subtitle: a NUL in "26\\x00degC" (meant 26°C) made the muxer stop
    # reading the SRT there and 166 subtitles were missing from the MKV (2026-10-17 day file, 13:34:53)
    t = re.sub(r'\x00\s*deg\s*([CF])\b', r'°\1', t)
    t = re.sub(r'\s?\bdeg(?:rees?)?C\b', '°C', t)
    t = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', t)
    t = re.sub(r"\bNorth Korean\b", "Korean", t)
    t = t.replace('℃', '°C').replace('℉', '°F')
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
    "a presenter, poetry recitation or dialogue with music playing underneath, even when singing or a choir can be heard "
    "behind the voice: what is SPOKEN is speech. A line that reports facts (who did what, when, dates, numbers) is speech. "
    "When unsure, use 'speech'. "
    "SONGS: give one stretch per sung lyric line (usually 3-7 seconds), starting exactly when that line starts to be "
    "sung and ending when its last note ends (a long held note included), never two lines merged into one; the times "
    "follow the singing, never spread evenly. Include every line of every verse and refrain, also when it repeats. " + DPRK_STYLE)
GEMINI_SCHEMA = {"type": "OBJECT", "properties": {"segments": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "start": {"type": "NUMBER"}, "end": {"type": "NUMBER"}, "kind": {"type": "STRING"}, "lang": {"type": "STRING"}, "ko": {"type": "STRING"}, "en": {"type": "STRING"}},
    "required": ["start", "end", "kind", "ko", "en"]}}}, "required": ["segments"]}


class Gemini:
    def __init__(self, key):
        self.key = key
        self.korean = {}                                   # English line -> the Korean Gemini heard
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
        try:
            for round_ in range(8):
                left = deadline - time.time()
                if left < 6:
                    return
                body = json.dumps({"contents": contents, "tools": tools, "generationConfig": {"temperature": 0}}).encode()
                req = gemini_request(self.key, GEMINI_MODEL, body)
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
        req = gemini_request(self.key, GEMINI_MODEL, body)
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
            req = gemini_request(self.key, model, body)
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
                    if len(self.korean) > 20000:
                        self.korean.clear()               # live runs for days: keep it bounded
                    self.korean[out[-1][2]] = ko          # kept so a recording can be re-timed later from the words
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
def gemini_json(key, parts, schema=None, timeout=120, tries=8, model=None):
    """One Gemini request returning JSON (retries while Google is busy). None if it never answered."""
    cfg = {"temperature": 0, "responseMimeType": "application/json"}
    if schema:
        cfg["responseSchema"] = schema
    body = json.dumps({"contents": [{"parts": parts}], "generationConfig": cfg}).encode()
    for attempt in range(tries):
        try:
            r = json.load(urllib.request.urlopen(gemini_request(key, model or GEMINI_MODEL, body), timeout=timeout))
            return json.loads(r['candidates'][0]['content']['parts'][0]['text'])
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 402, 403):
                return None
            time.sleep(3 + 2 * attempt)
        except Exception:
            time.sleep(3)
    return None


SONG_CHECK_MODEL = 'gemini-3.5-flash'     # only for song_check (user, 2026-10-17); everything else is Flash-Lite
SONG_CHECK_SCHEMA = {"type": "OBJECT", "properties": {"lines": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "n": {"type": "INTEGER"}, "how": {"type": "STRING", "enum": ["sung", "spoken", "not heard"]}}, "required": ["n", "how"]}}},
    "required": ["lines"]}


def song_check(key, x, lines, status=None, deadline=None):
    """Lines marked as a song are checked by a stronger model, line by line: sung, spoken or not heard. Flash-Lite
    takes narration or dialogue over music for lyrics and invents lyrics over instrumentals; on the reported cases
    (2026-10-17: 4:36 drama, 11:11 narration, 13:42 orchestra, 13:42 real song) Flash-Lite judged 8 of 15 right,
    3.5 Flash 14 of 15. Spoken -> shown as speech; not heard -> dropped; sung -> stays a lyric. Only pieces with a song
    line are sent. lines: [(start, end, english)] in seconds into x. Any failure keeps the lines as they were."""
    songs = [i for i, l in enumerate(lines) if l[2].startswith('♪')]
    # and spoken lines with no speech under them: lyrics Flash-Lite did not mark as a song, or words read off the
    # screen over music (2026-10-17 day file: 154 such subtitles, 1.9%) -- decided by the voice alone (below)
    silent = []
    if _vad is not None and any(l[1] - l[0] >= 1.2 and not l[2].startswith('♪') for l in lines):
        try:
            mix = _vad(np.asarray(x, dtype=np.float32), _VadOptions(min_silence_duration_ms=150, speech_pad_ms=0, threshold=0.5))
        except Exception:
            mix = None
        if mix is not None:
            silent = [i for i, (st, et, en) in enumerate(lines) if et - st >= 1.2 and not en.startswith('♪') and
                      sum(max(0.0, min(r['end'] / 16000, et + 0.3) - max(r['start'] / 16000, st - 0.3)) for r in mix) < 0.05 * (et - st)]
    if not key or not (songs or silent):
        if status is not None: status['ok'] = True
        return lines
    items = [{'n': k + 1, 'from_s': round(lines[i][0], 1), 'to_s': round(lines[i][1], 1), 'english': lines[i][2].lstrip('♪ ')}
             for k, i in enumerate(songs)]
    prompt = ("Listen to this audio. For each subtitle line below (with the time it is shown, in seconds from the start "
              "of the audio), say how its words are delivered: 'sung' if a voice SINGS them to a melody (a song, a choir "
              "singing words); 'spoken' if a voice SPEAKS them, even with music playing underneath (narration, a "
              "presenter, dialogue, poetry recitation); 'not heard' if no voice says or sings these words there (only "
              "instrumental music, or a choir humming without words). The times come from a rough transcription: if "
              "the words are sung or spoken anywhere in this audio, even at a different time, answer by how they are "
              "delivered; 'not heard' only if nobody sings or says them at all. "
              "Return {\"lines\": [{\"n\", \"how\"}]}.\n"
              + json.dumps(items, ensure_ascii=False))
    def ask():
        for attempt in range(2):                         # an incomplete answer is asked again once
            v = gemini_json(key, [{"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(to_flac(x)).decode()}},
                                  {"text": prompt}], SONG_CHECK_SCHEMA, model=SONG_CHECK_MODEL, tries=3)
            got = {d.get('n'): d.get('how') for d in (v or {}).get('lines', []) if isinstance(d, dict)}
            if len(got) >= len(songs):
                return got
        return {}
    got = ask() if songs else {}
    # The answer for narration over music changes from one asking to the next (11:11 on 2026-10-17: spoken, then
    # sung). If any line came back sung it is asked once more, and a line either answer hears as spoken is speech.
    left = lambda: float('inf') if deadline is None else deadline - time.time()
    if songs and len(got) >= len(songs) and 'sung' in got.values() and left() > 60:
        again = ask()
        for n, how in again.items():
            if how == 'spoken' and got.get(n) == 'sung':
                got[n] = 'spoken'
    if len(got) < len(songs):
        if status is not None: status['ok'] = False      # the caller must not keep this as the answer
        return lines
    if status is not None: status['ok'] = True
    out, drop, spoken = list(lines), set(), 0
    for k, i in enumerate(songs):
        how = got.get(k + 1)
        if how == 'spoken':
            out[i] = (lines[i][0], lines[i][1], lines[i][2].lstrip('♪ ').strip()); spoken += 1
        elif how == 'not heard':
            drop.add(i)
    # The model's word is not final: it calls narration over music sung and invents lyrics over a wordless choir,
    # and changes its mind between askings (2026-10-17). Every line still marked sung, and every 'not heard', is
    # decided by the separated voice (voice_kind); without demucs the model's answer stands.
    # the whole piece is separated, not just the song lines: demucs's output depends on the stretch it is given, and
    # cut to the lines it moved a borderline choir over the syllable threshold (2026-10-17 13:42:23)
    try:
        v = voice_track(x) if left() > len(x) / 16000 + 10 else None   # ~0.9 s a second of audio on one core
    except Exception as e:
        log('song check: voice separation failed: %s' % e); v = None
    if v is not None:
        try:
            ts = _vad(v, _VadOptions(min_silence_duration_ms=150, speech_pad_ms=0, threshold=0.5)) if _vad else []
        except Exception:
            ts = []
        drop = set()
        for i in songs:
            if out[i][2].startswith('♪') or got.get(songs.index(i) + 1) == 'not heard':
                kind, _ = voice_kind(x, v, ts, lines[i][0], lines[i][1])
                if kind == 'speech' and out[i][2].startswith('♪'):
                    out[i] = (lines[i][0], lines[i][1], lines[i][2].lstrip('♪ ').strip()); spoken += 1
                elif kind == 'none':
                    drop.add(i)
        sung = 0
        for i in silent:
            kind, _ = voice_kind(x, v, ts, lines[i][0], lines[i][1])
            if kind == 'sung':
                out[i] = (lines[i][0], lines[i][1], '♪ ' + lines[i][2]); sung += 1
            elif kind == 'none':
                drop.add(i)
        if silent:
            log('song check: of %d spoken lines with no speech under them, %d are sung, %d have no voice' % (len(silent), sung, len([i for i in silent if i in drop])))
    if spoken or drop:
        log('song check: %d of %d song lines are speech, %d not heard' % (spoken, len(songs), len(drop)))
    out = [l for i, l in enumerate(out) if i not in drop]
    if v is not None and any(l[2].startswith('♪') for l in out):
        out = time_songs(out, v)                         # lyrics on the singer, not on Gemini's rough times
    if status is not None: status['timed'] = v is not None
    return out


_DEMUCS, _DEMUCS_LOCK = None, threading.Lock()
_DEMUCS_SLOTS = threading.BoundedSemaphore(int(os.environ.get('DEMUCS_JOBS', '1')))   # separations at once


def voice_track(x):
    """The voice separated from the music (demucs htdemucs), at 16 kHz like x; None without demucs."""
    global _DEMUCS
    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from scipy.signal import resample_poly
    except Exception:
        return None
    with _DEMUCS_LOCK:
        if _DEMUCS is None:
            _DEMUCS = get_model('htdemucs'); _DEMUCS.eval()
    with _DEMUCS_SLOTS:                                  # live: one at a time, one core, lowest priority: the encoders come first
        tid = threading.get_native_id()
        try:
            was = os.getpriority(os.PRIO_PROCESS, tid); os.setpriority(os.PRIO_PROCESS, tid, 19)
        except Exception:
            was = None
        try:
            torch.set_num_threads(int(os.environ.get("DEMUCS_THREADS", "1")))   # 1: the work stays on this (niced) thread
            y = torch.tensor(resample_poly(np.asarray(x, dtype=np.float32), 441, 160).astype(np.float32)).unsqueeze(0).repeat(2, 1).unsqueeze(0)
            with torch.no_grad():
                out = apply_model(_DEMUCS, y, split=True, overlap=0.1, progress=False)[0]
            return resample_poly(out[_DEMUCS.sources.index('vocals')].mean(0).numpy(), 160, 441).astype(np.float32)[:len(x)]
        finally:
            if was is not None:
                try:
                    os.setpriority(os.PRIO_PROCESS, tid, was)
                except Exception:
                    pass


def sung_phrases(v):
    """Sung phrases in a separated voice track v: [(start, end)] in seconds. A phrase starts where the voice rises to
    near its singing level (within 12 dB of the piece's loud level) after being under it for 0.15 s+, or climbs back
    out of a breath (a dip 7 dB+ under both sides); it ends where the voice falls under that level for 0.15 s+."""
    n = len(v) // 160
    if n < 50:
        return []
    db = 20 * np.log10(np.sqrt((np.asarray(v[:n * 160], dtype=np.float32).reshape(n, 160) ** 2).mean(axis=1)) + 1e-9)
    sm = np.convolve(db, np.ones(5) / 5, 'same')
    loud = np.percentile(sm, 90)
    on = sm > loud - 12
    k = 0
    runs = []
    while k < n:                                         # runs of singing, closing gaps under 0.15 s
        if not on[k]:
            k += 1; continue
        j = k
        while j < n and (on[j] or (j + 15 < n and on[j:j + 15].any())):
            j += 1
        runs.append([k, j]); k = j
    out = []
    for a, b in runs:                                    # split a run at breaths
        cuts = [a]
        for i in range(a + 30, b - 30):
            if sm[i] == sm[i - 5:i + 6].min() and min(sm[i - 30:i].max(), sm[i:i + 30].max()) - sm[i] >= 7:
                top = sm[i:i + 30].max(); j = i
                while j < b - 1 and sm[j] < top - 4:
                    j += 1
                if j - cuts[-1] >= 40:
                    cuts.append(j)
        cuts.append(b)
        out += [(cuts[m] / 100, cuts[m + 1] / 100) for m in range(len(cuts) - 1) if cuts[m + 1] - cuts[m] >= 30]
    return out


def time_songs(lines, v):
    """Lyric lines (start, end, '♪ ...') put on the singer: each start on the nearest phrase start within 1.5 s of
    Gemini's (it is 0.5-1.3 s off either way: 2026-10-17 music video at 2:57), each end on the nearest phrase end within
    1.5 s of Gemini's; never before the line before has started, never past the next line. Spoken lines unchanged."""
    ph = sung_phrases(v)
    if not ph:
        return lines
    starts, ends = np.array([a for a, _ in ph]), np.array([b for _, b in ph])
    out, prev = list(lines), -1.0
    idx = [i for i, l in enumerate(lines) if l[2].startswith('♪')]
    for n, i in enumerate(idx):
        st, et, en = lines[i]
        c = starts[(starts >= max(st - SONG_SNAP_S, prev + 0.6)) & (starts <= st + SONG_SNAP_S)]
        s2 = float(c[np.argmin(np.abs(c - st))]) if len(c) else st
        e = ends[(ends >= et - 1.5) & (ends <= et + 1.5) & (ends > s2 + 0.5)]
        e2 = float(e[np.argmin(np.abs(e - et))]) if len(e) else max(et, s2 + 0.5)
        out[i] = (s2, e2, en); prev = s2
    for n in range(len(idx) - 1):
        a, b = out[idx[n]], out[idx[n + 1]]
        if a[1] > b[0] - 0.08:
            out[idx[n]] = (a[0], max(a[0] + 0.3, b[0] - 0.08), a[2])
    return out


CAPTION_MODEL = 'gemini-3.5-flash'   # reads KCTV's stylised lyric font; Flash-Lite misread it (훨한 for 험한), user 2026-10-18
CAPTION_MARK = '\u2063'             # (invisible) a line copied from an on-screen caption; rows joined by U+2028
CAPTION_STILL = 3.0                  # picture motion under this is a still (test card): see dayjob
CAPTION_MAX_ROW = 22                 # Korean letters: 1,170 of 1,174 lyric rows had 20 or fewer; documentary text 24+
CAPTION_MIN_S = 2.5                  # verses stay up 5 s+; a shorter caption was a title card (2026-10-17 13:42:23)
CAPTION_SCHEMA = {"type": "OBJECT", "properties": {"frames": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "t": {"type": "NUMBER"}, "text": {"type": "STRING"}}, "required": ["t", "text"]}}}, "required": ["frames"]}


def read_captions(key, video, t0, t1, fps=2.0, batch=40):
    """The song-lyric captions burned into a stretch of video (KCTV music videos write the Korean lyrics on screen,
    karaoke style: each syllable appears as it is sung). Frames of the lower half of the screen (the lyrics can be
    anywhere there, user 2026-10-18) every 1/fps s, read by Gemini. Returns [(t, text)] in the video's seconds, text
    with one line per row; None if reading failed."""
    r = subprocess.run(['nice', '-n', '19', 'ffmpeg', '-v', 'quiet', '-ss', '%.3f' % max(0.0, t0), '-i', video, '-t', '%.3f' % (t1 - t0),
                        '-vf', 'yadif,fps=%g,crop=iw:ih/2:0:ih/2,scale=640:-2' % fps, '-q:v', '4', '-f', 'image2pipe',
                        '-vcodec', 'mjpeg', '-'], capture_output=True, timeout=300).stdout
    jpgs = [b'\xff\xd8' + q for q in r.split(b'\xff\xd8')[1:]]
    out = []
    for b0 in range(0, len(jpgs), batch):
        parts = []
        for k in range(b0, min(len(jpgs), b0 + batch)):
            parts += [{"text": "t=%.1f" % (k / fps)}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpgs[k]).decode()}}]
        parts.append({"text": "These are frames, at the times given, of the lower half of the screen of a North Korean TV music video. "
                      "For each frame give the Korean SONG LYRIC caption exactly as written, with a line break (\\n) between its rows. "
                      "Only sung lyrics: ignore song titles, credits, names, channel logos, clocks and any other text. Empty text if "
                      "no lyric caption is on screen. Return {\"frames\": [{\"t\", \"text\"}]} with every frame."})
        v = gemini_json(key, parts, CAPTION_SCHEMA, model=CAPTION_MODEL, tries=3)
        got = {round(float(f.get('t', -1)), 1): str(f.get('text') or '').strip() for f in (v or {}).get('frames', []) if isinstance(f, dict)}
        if len(got) < 0.9 * (min(len(jpgs), b0 + batch) - b0):
            log('captions: %d of %d frames read at %.0f s' % (len(got), min(len(jpgs), b0 + batch) - b0, t0 + b0 / fps))
            return None
        out += [(t0 + k / fps, got.get(round(k / fps, 1), '')) for k in range(b0, min(len(jpgs), b0 + batch))]
    return out


def picture_motion(video, t0, t1):
    """How much the picture moves over t0..t1 (mean change between frames 0.5 s apart, small grey frames): the
    KCTV test card measures 0.1-1.6 (its clock), music videos 11-19. None if it could not be read."""
    r = subprocess.run(['nice', '-n', '19', 'ffmpeg', '-v', 'quiet', '-ss', '%.3f' % max(0.0, t0), '-i', video, '-t', '%.3f' % (t1 - t0),
                        '-vf', 'yadif,fps=2,scale=160:90,format=gray', '-f', 'rawvideo', '-'], capture_output=True, timeout=300).stdout
    n = len(r) // (160 * 90)
    if n < 3:
        return None
    d = np.frombuffer(r[:n * 160 * 90], np.uint8).reshape(n, 90, 160).astype(np.float32)
    return float(np.abs(np.diff(d, axis=0)).mean())


def caption_verses(frames):
    """Caption frames -> verses [(start, end, [korean rows])]: a verse starts when its caption appears and ends when
    it is cleared or replaced. (The captions wipe in over a few seconds, both rows together: when the caption appears
    is when the verse is sung, not when each row is, user check 2026-10-18 on the music video at 2:57.)"""
    out, verse, start, full = [], None, None, []
    dt = frames[1][0] - frames[0][0] if len(frames) > 1 else 0.5
    for t, text in frames + [(frames[-1][0] + dt, '')] if frames else []:
        r = [q.strip() for q in text.split('\n') if q.strip()]
        if verse is not None and (not r or r[0][:1] != verse[0][:1]):
            out.append((start, t - dt / 2, [q for q in full if q]))
            verse = None
        if r:
            if verse is None:
                start, full = t - dt / 2, []
            verse = r
            for k, q in enumerate(r):
                if k >= len(full):
                    full.append(q)
                elif len(q) >= len(full[k]):
                    full[k] = q
    return [v for v in out if v[2]]


def translate_lyrics(key, rows):
    """Korean lyric rows -> English subtitle lines, one per row, in DPRK house style."""
    if not rows:
        return []
    sch = {"type": "OBJECT", "properties": {"lines": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["lines"]}
    v = gemini_json(key, [{"text": "Translate these Korean song lyric lines from North Korean television into English subtitles, "
                                   "one English line per Korean line, in the same order, natural and singable, short. " + DPRK_STYLE +
                                   "\nReturn {\"lines\": [...]} with exactly %d lines.\n" % len(rows) +
                                   '\n'.join('%d. %s' % (i + 1, r) for i, r in enumerate(rows))}], sch, tries=3)
    out = [str(x).strip() for x in (v or {}).get('lines', [])]
    return [dprk_english(x) for x in out] if len(out) == len(rows) else None


def lyrics_from_captions(lines, verses, english, lo, hi):
    """Replace the lyric lines Gemini heard with the on-screen captions, for verses starting in lo..hi (the lines'
    clock). english: one line per row of the verses, in order. Each caption is copied as it is: one subtitle with its
    rows, up exactly while the caption is (user, 2026-10-18: "just replicate the on-screen text" -- captions are
    sometimes karaoke, sometimes static). In a piece with lyric captions, a lyric heard only in the sound is dropped:
    a music video puts every sung line on screen (2026-10-17 13:42:23)."""
    out, k, new, spans = list(lines), 0, [], []
    for vs, ve, rows in verses:
        en = english[k:k + len(rows)]; k += len(rows)
        if not all(en) or ve < lo or vs >= hi or ve - vs < CAPTION_MIN_S:
            continue                                     # (a caption up for a moment is a title card, not a verse)
        if max(len(HANGUL.findall(r)) for r in rows) > CAPTION_MAX_ROW:
            continue                                     # a sentence of on-screen text (a documentary), not a lyric
        spans.append((vs - 1.0, ve + 0.5))
        if lo <= vs < hi:
            new.append((vs, ve, '♪ ' + CAPTION_MARK + '\u2028'.join(e.strip() for e in en)))
    if not spans:
        return lines
    kept = [l for l in out if not l[2].startswith('♪')]
    return sorted(kept + new)


_WHISPER, _WHISPER_LOCK = None, threading.Lock()


def whisper_words(x):
    """Word times heard by Whisper (faster-whisper "small", on this machine, free) in x: [(word, start, end)] in
    seconds into x; None without it. Used for where lines END: judged against Speech-to-Text alone (2026-10-18),
    ends before a pause within 0.3 s went 88% -> 94%, within 0.5 s 91% -> 98% (its starts measured poor)."""
    global _WHISPER
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    with _WHISPER_LOCK:
        if _WHISPER is None:
            _WHISPER = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=int(os.environ.get('WHISPER_THREADS', '2')),
                                    num_workers=int(os.environ.get('WHISPER_JOBS', '1')), download_root='/DATA/subs/tools/models')
    segs, _ = _WHISPER.transcribe(np.asarray(x, dtype=np.float32), language='ko', word_timestamps=True, vad_filter=False,
                                  condition_on_previous_text=False, beam_size=1)
    return [(w.word.strip(), float(w.start), float(w.end)) for sg in segs for w in (sg.words or []) if w.word.strip()]


def voice_kind(x, v, ts, st, et):
    """('speech' | 'sung' | 'none', numbers) for a line at st..et of x (widened 1 s: the times are Gemini's), from its
    separated voice v and Silero runs ts on v. Measured 2026-10-17: narration over music is talk on the voice track
    (0.42-0.83 of the line; real lyrics 0.00); lyrics invented over instrumentals have no voice (under 10 % of
    frames); a wordless choir under a fanfare (13:42:20-52) is a voice held flat 4-6 dB under the music with no
    syllables, while a sung lead is within ~3 dB of it or moves in syllables."""
    st, et = max(0.0, st - 1), min(len(x) / 16000, et + 1)
    a, b = int(st * 16000), int(et * 16000)
    n = (b - a) // 1600
    if n < 5:
        return 'sung', None
    db = lambda z: 10 * np.log10((np.asarray(z[a:a + n * 1600], dtype=np.float32).reshape(n, 1600) ** 2).mean(axis=1) + 1e-12)
    dv, dx = db(v), db(x)
    on = (dv - dx) > -10
    voice = float(on.mean())
    level = float(np.median((dv - dx)[on])) if on.any() else -60.0
    m = (b - a) // 160
    env = 10 * np.log10((np.asarray(v[a:a + m * 160], dtype=np.float32).reshape(m, 160) ** 2).mean(axis=1) + 1e-12)
    try:
        from scipy.signal import find_peaks
        syl = len(find_peaks(np.convolve(env, np.ones(5) / 5, 'same'), prominence=6, distance=10)[0]) / (m / 100)
    except Exception:
        syl = 1.0
    talk = sum(max(0.0, min(t['end'] / 16000, et) - max(t['start'] / 16000, st)) for t in ts) / max(0.1, et - st)
    nums = (round(voice, 2), round(level, 1), round(syl, 2), round(talk, 2))
    if talk >= VOICE_TALK_MIN:
        return 'speech', nums
    if voice < SONG_VOICE_MIN or (level < SONG_LEVEL_MIN and syl < SONG_SYLLABLES_MIN):
        return 'none', nums
    return 'sung', nums


def speech_or_music(key, flac, model=None):
    """(speech, singing) for a stretch of audio: a chunk that came back empty but is full of sound is usually music,
    sometimes speech Gemini skipped (2026-10-16: 144 music, 25 speech among such chunks)."""
    v = gemini_json(key, [{"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(flac).decode()}},
                          {"text": "Is anyone speaking in this audio? Is anyone singing words?"}],
                    {"type": "OBJECT", "properties": {"speech": {"type": "BOOLEAN"}, "singing": {"type": "BOOLEAN"}},
                     "required": ["speech", "singing"]}, model=model)
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
        # shorter by enough (to the target, or by 15%+); every number kept; names kept (80%+ of the capitalised
        # words: in KCTV English most words are capitalised, "Party Central Committee", and all-or-nothing refused
        # nearly every shortening, 2026-10-17)
        if not new or (len(new) > mx + 3 and len(new) > 0.85 * len(old)):
            return False
        caps = re.findall(r'\b[A-Z][a-z]+\b', old)[1:]
        return all(n in new for n in re.findall(r'\d+', old)) and (not caps or sum(c in new for c in caps) >= 0.8 * len(caps))
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
    t = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', t)            # every writer passes here: no control characters
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


ABBREV = {'Mt.', 'St.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'No.', 'Prof.', 'Gen.', 'Col.', 'Lt.', 'Capt.'}


def in_leader(words, i):
    """True if a break before words[i] would cut a leader's name (Kim Il Sung, Kim Jong Il, Kim Jong Un, Kim Jong
    Suk): never allowed."""
    if i <= 0 or i >= len(words):
        return False
    left = ' '.join(words[max(0, i - 3):i])
    for m in LEADERS.finditer(' '.join(words[max(0, i - 3):i + 3])):
        if m.start() < len(left) < m.end():
            return True
    return False


def in_name(words, i):
    """True if a break before words[i] would cut a name in two (user: names are never split over lines or
    subtitles): both sides capitalised and nothing that ends a phrase in between ("Kim | Jong Un", "Ri Il | Hwan",
    "Mt. | Paektu", "Sinhung | County"). A leader's name is caught whatever the case of its parts."""
    if i <= 0 or i >= len(words):
        return False
    a, b = words[i - 1], words[i].lstrip('"“‘(')
    if in_leader(words, i):
        return True
    if a in ABBREV:
        return b[:1].isupper()
    if a.endswith((',', ';', ':', '.', '?', '!', '"', '”', ')')) or a in ('I', 'A'):
        return False
    return a[:1].isupper() and b[:1].isupper() and a[:1].isalpha() and b[:1].isalpha()


def split_two(text):
    """One subtitle's text as 1 or 2 balanced lines of at most LINE characters (None if it does not fit)."""
    if len(text) <= LINE:
        return [text]
    words, best = text.split(' '), None
    for i in range(1, len(words)):
        a, b = ' '.join(words[:i]), ' '.join(words[i:])
        if len(a) > LINE or len(b) > LINE or in_leader(words, i):
            continue
        score = abs(len(a) - len(b)) + (1000 if in_name(words, i) else 0)   # inside a name: only if nothing else fits
        last = words[i - 1].lower().strip(',;:')
        if words[i - 1].endswith((',', ';', ':')): score -= 14            # break after a comma
        if words[i].lower() in BREAK_BEFORE: score -= 8                     # break before a conjunction
        if last in WEAK_END: score += 25                                    # never leave "the"/"of" hanging
        if i == 1 or i == len(words) - 1: score += 30                       # no line of a single word
        if best is None or score < best[0]:
            best = (score, [a, b])
    if best and any(' ' not in x for x in best[1]) and len(text) <= LINE + 4:
        return [text]                                   # a line a little long beats a word on its own
    if best and best[0] >= 1000 and len(text) > 2 * LINE - 10 and NAME_SAFE_SPLIT:
        return None                                     # only a break inside a name fits: two subtitles instead
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
            full = cur + [w]
            k = cut                                        # not between two parts of a name: the name moves on
            while k > 1 and k > cut - 5 and in_name(full, k):
                k -= 1
            if not in_name(full, k) and len(' '.join(cur[:k])) >= 12:
                cut = k
            while cut > 1 and in_leader(full, cut):        # a leader's name is never cut, whatever it costs
                cut -= 1
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
            if not in_name(both, i) and split_two(a) and split_two(b):
                score = abs(len(a) - len(b)) - (12 if both[i - 1].endswith((',', ';', ':')) else 0)
                if best is None or score < best[0]:
                    best = (score, a, b)
        if best:
            out[-2:] = [best[1], best[2]]
    return out


LINGER_S = 0.2                                      # a subtitle stays at most this long after its words end
LATE_S = 0.1                                  # most a subtitle may start after its voice (see polish)
SPEECH_GAP_S = 1.5                                      # see speech_gaps
SONG_LEVEL_MIN = -3.5                                   # see voice_kind
SONG_SYLLABLES_MIN = 0.5                                # see voice_kind
VOICE_TALK_MIN = 0.3                                    # see voice_kind
SONG_VOICE_MIN = 0.1                                    # see voice_kind
SUNG_MAX_SPEECH = 0.5                                   # see spoken_not_sung
NAME_SAFE_SPLIT = True                                  # see split_two
READ_CPS, READ_MIN_S = 20.0, 1.0                       # reading speed and the shortest subtitle
LEAD_S = 0.0                                  # a subtitle never shows before its voice (user: 'start early')


def make_cues(lines, onsets=None, open_end=False):
    """Lines (start, end, english) -> finished subtitles (start, end, 'line1\nline2') following subtitle rules:
    songs dropped and overlaps tidied (tidy_lines), clean text, at most 2 balanced lines of 42, long lines split
    at clause breaks with time shared by length, at least 1 s / ~17 characters per second on screen (stretching
    into free time), at most 7 s, and an 80 ms gap so a change of subtitle is visible."""
    cues = []
    for st, et, en in tidy_lines(lines):
        song = en.startswith('♪')
        en = en.lstrip('♪ ').strip()
        if CAPTION_MARK in en:                               # an on-screen caption: one subtitle, its rows as they are
            rs = [r.strip() for r in en.replace(CAPTION_MARK, '').split('\u2028') if r.strip()]
            if len(rs) <= 2 and all(len(r) <= LINE for r in rs):
                cues.append([st, et, en, True, et, True]); continue
            # too long for two lines as on screen: the same words in two balanced lines, or two subtitles sharing the
            # caption's time by length
            joined = ' '.join(rs)
            if split_two(joined):
                cues.append([st, et, CAPTION_MARK + '\u2028'.join(split_two(joined)), True, et, True]); continue
            halves = split_cues(joined)
            if len(halves) < 2:
                w = joined.split(); halves = [' '.join(w[:len(w) // 2]), ' '.join(w[len(w) // 2:])]
            tot, t0 = sum(len(h) for h in halves) or 1, st
            for h in halves:
                t1 = t0 + (et - st) * len(h) / tot
                cues.append([t0, t1 - (0.08 if t1 < et else 0), CAPTION_MARK + '\u2028'.join(split_two(h) or [h]), True, t1, True])
                t0 = t1
            continue
        # capital only at the start of the sentence; a continuation keeps its lower case
        # a sentence split over several subtitles: the middle parts keep a natural comma, no dash or semicolon
        parts = [re.sub(r'[\s;:\-–—]+$', '', p).strip() for p in split_cues(clean_text(en) if not song else clean_text(en).rstrip('.'))]
        if song:                                             # lyrics: one line per subtitle, shown as it is sung
            parts = [r for p in parts for r in (split_two(p) or [p])]
        parts = [p for p in parts if re.search(r'[A-Za-z0-9]', p)]
        if not parts:
            continue
        tot = sum(len(p) for p in parts) or 1
        cut, done = [st], 0                                  # where each part starts: by its share of the characters,
        for p in parts[:-1]:                                 # then moved to the speaker's pause nearby (onsets: times
            done += len(p)                                   # the voice resumes, same clock as the lines). Each from
            t = st + (et - st) * done / tot                  # the line's own start: a snap must not carry into the
                                                             # next (2026-10-18 news 0:45: a part ran past the line)
            if onsets is not None and len(onsets) and not song:
                on = np.asarray(onsets)
                ot = on[:, 0] if on.ndim == 2 else on
                inside = (ot >= cut[-1] + 0.8) & (ot <= et - 0.6)
                ok = inside & (np.abs(ot - t) <= 0.8)
                if on.ndim == 2:                         # a real pause (0.2 s+) is looked for further: 1.5 s
                    ok |= inside & (np.abs(ot - t) <= 1.5) & (on[:, 1] >= 0.2)
                if ok.any():
                    if on.ndim == 2 and on[ok, 1].max() >= 0.12:
                        # the speaker's break: the longest quiet before a restart (2026-10-17 day file: placed by
                        # length the second part came 0.4-1.4 s after its words, at a gap between syllables)
                        c = on[ok]
                        t = float(c[np.argmax(c[:, 1] - 0.1 * np.abs(c[:, 0] - t)), 0])
                    else:
                        c = ot[ok]
                        t = float(c[np.argmin(np.abs(c - t))])
            t = min(max(t, cut[-1] + 0.5), et - 0.5)       # in order, inside the line
            cut.append(t)
        cut.append(et)
        for k, p in enumerate(parts):                        # [4]: where the words end; [5]: starts on the voice
            cues.append([cut[k], cut[k + 1], p, song, cut[k + 1], k == 0])
    cues.sort(key=lambda c: c[0])
    for i, c in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else c[0] + 7.0
        want = max(1.0, len(c[2]) / READ_CPS)
        c[1] = max(c[1], min(c[0] + want, nxt - 0.08))                     # long enough to read, if there is room
        if CAPTION_MARK in c[2]:
            # copied from the screen: up while the caption is up (user, 2026-10-18: "just replicate the on-screen
            # text"), only cut where the next subtitle begins
            c[1] = min(c[1], nxt - 0.08) if i + 1 < len(cues) else c[1]
            continue
        if c[3]:
            # lyrics keep their times (on-screen captions, or timed on the singer): only cut short where the next
            # subtitle begins, readable if there is room, at most 7 s
            c[1] = min(c[4], nxt - 0.08, c[0] + 7.0) if i + 1 < len(cues) else min(c[4], c[0] + 7.0)
            c[1] = max(c[1], min(c[0] + 0.83, nxt - 0.08) if i + 1 < len(cues) else c[0] + 0.83)
            continue
        c[1] = min(c[1], c[0] + 7.0, nxt - 0.08 if i + 1 < len(cues) else c[1])
        # ends with the words (+0.4 s), but never before it can be read (~17 characters a second, at least 1.2 s):
        # it then stays on into the free time after, up to the next subtitle (user, 2026-10-17: "subs only show for
        # like a second, I can't read them")
        c[1] = min(max(c[4] + LINGER_S, c[0] + max(READ_MIN_S, len(c[2]) / READ_CPS)), c[0] + 7.0)
        if i + 1 < len(cues):
            c[1] = min(c[1], cues[i + 1][0] - 0.08)
        elif open_end:
            # live: what follows is in the next batch, which can't overlap a subtitle already published and would
            # push its first line late (2026-10-17 waveform check: 27% of starts late) -> no reading extension here
            c[1] = min(c[1], max(c[4] + LINGER_S, c[0] + READ_MIN_S))
        c[1] = max(c[1], c[0] + 0.83)                                       # broadcast minimum: 5/6 of a second
    cues = polish(cues)
    # last safety net: never a subtitle that ends before it starts or flashes for a moment (2026-10-17 day file
    # 2:17:33 had three ending before they began): joined to the next one if they fit together, else lengthened
    # into the free time after it, else dropped
    safe = []
    for i, c in enumerate(cues):
        if c[1] - c[0] >= 0.3:
            safe.append(c); continue
        n = cues[i + 1] if i + 1 < len(cues) else None
        joined = c[2] + ' ' + n[2] if n is not None else None
        if n is not None and n[3] == c[3] and split_two(joined) and len(joined) <= 2 * LINE and n[1] - c[0] <= 7.0:
            n[0], n[2] = c[0], joined
            continue
        room = (n[0] - 0.08 if n is not None else c[0] + 0.83) - c[0]
        if room >= 0.5:
            c[1] = c[0] + min(room, max(0.83, len(c[2]) / READ_CPS)); safe.append(c)
    cues = safe
    def rows(c):
        if CAPTION_MARK in c[2]:
            rs = [r.strip() for r in c[2].replace(CAPTION_MARK, '').split('\u2028') if r.strip()]
            if len(rs) == 1 and len(rs[0]) > LINE:          # one long row: two lines, as any subtitle
                rs = split_two(rs[0]) or rs
            return song_marks(rs)
        return song_marks(split_two(c[2]) or [c[2]]) if c[3] else '\n'.join(split_two(c[2]) or [c[2]])
    return [(c[0], c[1], rows(c)) for c in cues]


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
    fixed = lambda c: c[3]                               # a lyric keeps its times: nothing here moves it
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if fixed(a) or fixed(b):
            continue
        if a[1] - a[0] < 0.83 and b[1] - b[0] > 1.6:
            need = min(0.83 - (a[1] - a[0]), (b[1] - b[0]) - 1.2)
            if b[0] - a[1] < need + 0.08:
                b[0] = min(a[1] + need + 0.08, b[0] + late(b)) if b[0] < a[1] + need + 0.08 else b[0]
            a[1] = min(a[1] + need, b[0] - 0.08)
    # too much to read (over 20 characters a second): a small lead-in from free time before (at most 0.3 s, as on
    # broadcast TV), then time the next subtitle can spare (it starts a little later but keeps enough for its own text)
    for i, c in enumerate(out):
        need = len(c[2]) / READ_CPS - (c[1] - c[0])
        if need <= 0 or len(c[2]) / max(0.1, c[1] - c[0]) <= 20 or fixed(c):
            continue
        free_before = c[0] - (out[i - 1][1] + 0.08 if i else 0)
        lead = max(0.0, min(LEAD_S, free_before, need))
        c[0] -= lead; need -= lead
        if need > 0 and i + 1 < len(out):
            n = out[i + 1]
            spare = 0.0 if fixed(n) else (n[1] - n[0]) - max(0.83, len(n[2]) / READ_CPS)
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
        if need <= 0 or fixed(c):
            continue
        # never by starting earlier (user, 2026-10-17: "lines show before it is said"): only into free time after
        nxt = out[i + 1][0] - 0.08 if i + 1 < len(out) else c[1] + need
        c[1] = max(c[1], min(c[0] + 0.83, nxt))
    # fragments too short to read (a long sentence cut into pieces): join neighbours into one subtitle when they
    # follow on and together fit two lines and 7 s (2026-10-16 day file: 183 subtitles under 0.83 s)
    merged = []
    for c in out:
        if merged:
            p = merged[-1]
            short = (p[1] - p[0] < READ_MIN_S) or (c[1] - c[0] < READ_MIN_S)
            joined = p[2] + ' ' + c[2]
            if (short and p[3] == c[3] and not fixed(c) and c[0] - p[1] <= 0.3 and c[1] - p[0] <= 7.0 and split_two(joined)
                    and len(joined) <= 2 * LINE):
                p[1], p[2], p[4] = c[1], joined, c[4]
                continue
        merged.append(c)
    out = merged
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        if b[0] < a[1] + 0.08 and a[1] - a[0] <= 0.9 and b[1] - (a[1] + 0.08) >= 1.2 and a[1] + 0.08 - b[0] <= late(b) and not fixed(b):
            b[0] = a[1] + 0.08                      # a short line keeps its time; the long next one starts a moment later
        if out[i][0] < out[i - 1][1] + 0.08:
            out[i - 1][1] = max(out[i - 1][0] + 0.5, out[i][0] - 0.08)
            if out[i][0] < out[i - 1][1] + 0.08 and not fixed(out[i]):   # still overlapping: the next one starts a moment later
                out[i][0] = out[i - 1][1] + 0.08
                out[i][1] = max(out[i][1], out[i][0] + 0.5)
    return out


SONG_SNAP_S = 1.5          # see time_songs
SONG_MARKS = True           # user, 2026-10-18: music notes back (♪ ... ♪) now that lyrics are checked against the
                            # separated voice and taken from the on-screen captions; earlier the same day they were off


def song_marks(rows):
    """Sung lyrics: shown as they are (SONG_MARKS: a music note at the start of the first line and the end of the last)."""
    rows = list(rows)
    if SONG_MARKS:
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


LOCKED_16K = 'aresample=16000:async=1:min_hard_comp=0.02:first_pts=0'   # decode on the sound's own timestamps


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


def vocal_starts(x, depth=8.0):
    """Where a singer starts a phrase over music: the voice band's share of the sound (300-3400 Hz against the
    rest) dips between sung phrases (a breath) and climbs back. Times in s into x (16 kHz)."""
    hop = 320
    n = len(x) // hop
    if n < 40:
        return np.array([])
    fr = x[:n * hop].reshape(n, hop).astype(np.float32) * np.hanning(hop)
    X = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(hop, 1 / 16000)
    band, rest = (f > 300) & (f < 3400), (f < 250) | (f > 4000)
    r = 10 * np.log10(X[:, band].sum(axis=1) + 1e-9) - 10 * np.log10(X[:, rest].sum(axis=1) + 1e-9)
    r = np.convolve(r, np.ones(3) / 3, 'same')
    out, i = [], 20
    while i < n - 20:
        if r[i] == r[i - 5:i + 6].min():
            d = min(r[i - 20:i].max(), r[i:i + 20].max()) - r[i]
            if d >= depth:
                low, j = r[i] + d / 2, i
                while j < n - 1 and r[j] < low:
                    j += 1
                out.append(j * 0.02)
                i = j + 10
                continue
        i += 1
    return np.array(out)


def voice_starts(x, pauses=False):
    """Every place the voice starts or resumes in x: after silence (speech_onsets) or after a short pause (pause_ends).
    make_cues puts the break between two parts of a long line on one of these. pauses: rows of (time, seconds of quiet
    just before it, up to 1 s) -- these places come every 0.3-0.8 s (gaps between syllables too), and the speaker's
    real break between phrases is the one after the longest quiet."""
    t = np.unique(np.round(np.concatenate([speech_onsets(x), pause_ends(x)]), 2))   # (vocal_starts fired on every syllable)
    if not pauses:
        return t
    m = len(x) // 160
    if m < 10 or not len(t):
        return np.zeros((0, 2))
    db = 20 * np.log10(np.sqrt((x[:m * 160].reshape(m, 160).astype(np.float32) ** 2).mean(axis=1)) + 1e-9)
    live = db[db > -100]
    quiet = db < (np.percentile(live, 25) if len(live) else -60) + 6
    out = []
    for v in t:
        j = min(m, int(round(v * 100))) - 3                # the voice's own rise (a frame or two) is not counted
        k = j
        while k > 0 and j - k < 100 and quiet[k - 1]:
            k -= 1
        out.append((v, (j - k) / 100))
    return np.array(out)


def align_score(lines, x):
    """How well a chunk's spoken lines sit on the speech: the share whose start is within 0.3 s of a speech onset,
    and how many spoken lines there are (songs are not counted: singing over music has no clean onsets)."""
    spoken = [l for l in lines if not l[2].startswith('♪')]
    on = speech_onsets(x)
    if not spoken or not len(on):
        return 1.0, len(spoken)
    return sum(1 for st, et, en in spoken if float(np.min(np.abs(on - st))) <= 0.3) / len(spoken), len(spoken)


def well_aligned(gemini, flac, x, seconds, deadline, expect, lines, tries=2):
    """If fewer than 60% of a chunk's spoken lines start on speech (Gemini timed the chunk badly), ask Gemini again (up
    to `tries` times, while time allows) and keep the best-aligned answer. Returns Gemini's lines as given (timing is
    applied afterwards by snap_to_speech, so a stored answer can always be re-timed)."""
    best, raw = snap_to_speech(lines, x), lines
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
            best, raw, score, n = cand, new, cs, cn
    return raw


FOREIGN = re.compile(r'[^\x00-\x7F♪‘’“”—–…°·éèêáàâüöäïçñ]')


def fix_foreign(key, lines):
    """A line with letters of another script in its English (2026-10-16: "Let's produce zოკარო using milk", a garbled
    요구르트 = yoghurt): Gemini rewrites just that word in English, from the lines around it. Other lines untouched."""
    bad = [i for i, l in enumerate(lines) if FOREIGN.search(l[2].lstrip('♪ '))]
    if not bad or not key:
        return lines
    lines = list(lines)
    items = [{'n': i, 'before': lines[i - 1][2] if i else '', 'line': lines[i][2],
              'after': lines[i + 1][2] if i + 1 < len(lines) else ''} for i in bad]
    prompt = ("These English subtitles of DPRK television contain a word in the wrong script: a Korean word "
              "garbled in transcription. Replace only that word with the English word it most likely is (from the "
              "context: e.g. a dairy product made from milk is 'yoghurt'), keep every other word exactly. " + DPRK_STYLE +
              ' Return {"lines": [...]}, one string per item, in order.\n' + json.dumps(items, ensure_ascii=False))
    v = gemini_json(key, [{"text": prompt}], {"type": "OBJECT", "properties": {"lines": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["lines"]})
    out = (v or {}).get('lines') or []
    for i, new in zip(bad, out if len(out) == len(bad) else []):
        new = str(new).strip()
        if new and not FOREIGN.search(new) and abs(len(new.split()) - len(lines[i][2].split())) <= 2:
            log('foreign letters fixed: %r -> %r' % (lines[i][2], new))
            lines[i] = (lines[i][0], lines[i][1], ('♪ ' if lines[i][2].startswith('♪') and not new.startswith('♪') else '') + new) + tuple(lines[i][3:])
    return lines


def speech_gaps(lines, x, min_s=4.0):
    """Stretches (start, end) of x with speech but no line: whole seconds with 0.5 s+ of voice, not within 0.5 s of
    any line, at least min_s long. Gemini sometimes stops early (2026-10-16 12:55: the last 16 s of a chunk) or
    skips a stretch; music without a voice scores low and is not counted."""
    secs = int(len(x) / 16000)
    cover = np.zeros(secs, dtype=bool)
    for st, et, en in lines:
        cover[max(0, int(st - 0.5)):max(0, min(secs, int(np.ceil(et + 0.5))))] = True
    # what counts as a voice: the speech band's level (catches singing too, but also music: needs min_s) or, from
    # 2.5 s, the Silero speech detector (speech only, so a short line someone skipped is caught: 2026-10-17 4:03:59)
    speech = np.zeros(secs)
    if _vad is not None:
        try:
            for t in _vad(np.asarray(x, dtype=np.float32), _VadOptions(min_silence_duration_ms=150, speech_pad_ms=0, threshold=0.5)):
                for s_ in range(int(t['start'] / 16000), min(secs, int(np.ceil(t['end'] / 16000)))):
                    speech[s_] += min(s_ + 1, t['end'] / 16000) - max(s_, t['start'] / 16000)
        except Exception:
            pass
    gaps = []
    for test, need in ((lambda s_: voiced_seconds(x, s_, s_ + 1) >= 0.5, min_s), (lambda s_: speech[s_] >= 0.5, SPEECH_GAP_S)):
        run = None
        for s_ in range(secs):
            talk = not cover[s_] and test(s_)
            if talk and run is None:
                run = s_
            if (not talk or s_ == secs - 1) and run is not None:
                end = s_ + 1 if talk else s_
                if end - run >= need and not any(g0 < end and run < g1 for g0, g1 in gaps):
                    gaps.append((float(run), float(end)))
                run = None
    return sorted(gaps)


def fill_gaps(gemini, flac, x, seconds, deadline, expect, lines, tries=2):
    """Speech left without subtitles (speech_gaps): ask Gemini again for the chunk and take only the new lines that
    fall in the gaps and touch no line already there. Lines stay as Gemini gave them (timed later, like the rest)."""
    # first, each gap on its own: the gap's sound with a second either side. A short clip keeps Gemini on the words
    # it skipped (2026-10-17 music video: a repeated chorus stayed missing when the whole chunk was asked again)
    for g0, g1 in speech_gaps(lines, x)[:3]:
        if time.time() > deadline - 25:
            break
        a0, a1 = max(0.0, g0 - 1.0), min(len(x) / 16000, g1 + 1.0)
        new = gemini.subtitles(to_flac(x[int(a0 * 16000):int(a1 * 16000)]), a1 - a0, deadline, True)
        add = [(st + a0, et + a0, en) for st, et, en in (new or [])]
        add = [l for l in add if g0 - 1 <= (l[0] + l[1]) / 2 <= g1 + 1 and not any(l[0] < et and st < l[1] for st, et, en in lines)]
        if add:
            log('filled %d s of speech without subtitles with %d line(s) (gap asked alone)' % (g1 - g0, len(add)))
            lines = sorted(list(lines) + add)
    for attempt in range(tries):
        gaps = speech_gaps(lines, x)
        if not gaps or time.time() > deadline - 20:
            break
        new = gemini.subtitles(flac, seconds, deadline, expect)
        if not new:
            continue
        add = [l for l in new
               if any(g0 - 1 <= (l[0] + l[1]) / 2 <= g1 + 1 for g0, g1 in gaps)
               and not any(l[0] < et and st < l[1] for st, et, en in lines)]
        if add:
            log('filled %d s of speech without subtitles with %d line(s)' % (sum(g1 - g0 for g0, g1 in gaps), len(add)))
            lines = sorted(list(lines) + add)
    return lines


try:                                     # Silero speech detector (from faster-whisper): tells speech from music
    _venv = '/DATA/subs/tools/venv/lib/python%d.%d/site-packages' % sys.version_info[:2]
    if os.path.isdir(_venv) and _venv not in sys.path:
        sys.path.append(_venv)
    from faster_whisper.vad import get_speech_timestamps as _vad, VadOptions as _VadOptions
except Exception:
    _vad = None


def speech_ends(x):
    """Times (s into x, 16 kHz) where speech stops, by the Silero speech detector; empty without it."""
    if _vad is None or len(x) < 16000:
        return np.array([])
    try:
        ts = _vad(np.asarray(x, dtype=np.float32), _VadOptions(min_silence_duration_ms=VAD_SILENCE_MS, speech_pad_ms=0,
                                                              threshold=VAD_THRESHOLD))
    except Exception:
        return np.array([])
    return np.array([t['end'] / 16000 for t in ts])


VAD_SILENCE_MS, VAD_THRESHOLD = 150, 0.65               # Silero settings for speech_ends (0.65: better ends on both test sets)
HANGUL = re.compile(r'[가-힣0-9]')
END_TRIM_S = 0.1                                        # Silero stops a little late
WORD_END_SHIFT = float(os.environ.get('WORD_END_SHIFT', '0.05'))   # Whisper word ends, best against Speech-to-Text
START_BEFORE_S, START_AFTER_S = 0.8, 2.0                # pause onset search around Gemini's start (tested)


def stt_words(key, x, tries=3):
    """Korean words with times (s into x) from Google Speech-to-Text, or None (no key, or it failed). Used only for
    timing: which stretch of sound is which line (the text comes from Gemini)."""
    if not key or len(x) < 16000:
        return None
    pcm = (np.clip(x, -1, 1) * 32767).astype('<i2').tobytes()
    body = json.dumps({"config": {"encoding": "LINEAR16", "sampleRateHertz": 16000, "languageCode": "ko-KR",
                       "enableWordTimeOffsets": True, "model": "default"},
                       "audio": {"content": base64.b64encode(pcm).decode()}}).encode()
    for attempt in range(tries):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                'https://speech.googleapis.com/v1/speech:recognize', data=body,
                headers={'Content-Type': 'application/json', 'x-goog-api-key': key}), timeout=60))
            return [(w['word'], float(w['startTime'][:-1]), float(w['endTime'][:-1]))
                    for res in r.get('results', []) for w in res['alternatives'][0].get('words', [])]
        except Exception as e:
            log('Speech-to-Text failed: %s' % str(e)[:80])
            time.sleep(2 * (attempt + 1))
    return None


def word_span(ko, words, lo, hi, share=0.3):
    """(start, end) of a line's Korean among recognised words between lo and hi, letter by letter; an edge is None
    unless the line's first/last letters themselves were matched; None if under half the line matched."""
    ch, tm = [], []
    for w, a, b in words:
        if lo <= a <= hi:
            cs = HANGUL.findall(w)
            for i, c in enumerate(cs):
                ch.append(c); tm.append((a + (b - a) * i / len(cs), a + (b - a) * (i + 1) / len(cs)))
    k = ''.join(HANGUL.findall(ko or ''))
    if len(k) < 4 or not ch:
        return None
    blocks = [b for b in difflib.SequenceMatcher(None, k, ''.join(ch), autojunk=False).get_matching_blocks() if b.size >= 2]
    if sum(b.size for b in blocks) < share * len(k):
        return None
    st = tm[blocks[0].b][0] if blocks[0].a <= 2 else None
    en = tm[blocks[-1].b + blocks[-1].size - 1][1] if blocks[-1].a + blocks[-1].size >= len(k) - 2 else None
    return st, en


def quiet_onsets(x, tolerant=False):
    """Where speech comes back after at least 0.3 s of quiet (10 ms frames, 10 dB over the floor)."""
    m = len(x) // 160
    if m < 60:
        return np.array([])
    db = 20 * np.log10(np.sqrt((x[:m * 160].reshape(m, 160).astype(np.float32) ** 2).mean(axis=1)) + 1e-9)
    live = db[db > -100]
    fl = np.percentile(live, 25) if len(live) else -60
    quiet = db < fl + 6
    # quiet frames in the 30 before; tolerant: 30 of the 35 before, so the voice's own rise (a frame or two between
    # quiet and voiced) can't hide its start (2026-10-17 news 3:45: a -32 dB frame between -42 and -25 dB). The
    # tolerant list only serves the look-back in timed_lines: as the main search it found too many (84% vs 97%).
    w = 35 if tolerant else 30
    run = np.convolve(quiet.astype(int), np.ones(w, dtype=int), 'full')[:m]
    # and the sound must carry on (half of the next 0.25 s): a short noise in a quiet scene is not the voice coming
    # back (2026-10-17 4:04:28: a line snapped 1.4 s early onto a faint knock)
    on = [j / 100 for j in range(30, m - 1) if db[j] > fl + 10 and run[j - 1] >= 30 and (db[j:j + 25] > fl + 10).mean() >= 0.5]
    # a piece that begins inside a pause (pieces are cut at quiet moments): its first voice counts too, if all of
    # the 0.1 s+ before it in the piece is quiet (2026-10-17 news 3:45: the start was missed and shown 1.4 s late)
    first = next((j for j in range(10, min(m - 1, 150)) if db[j] > fl + 10), None)   # up to 1.5 s in (2026-10-18:
    # 20:00 news 26:36, speech 0.7 s into a piece was missed and the line snapped 1.7 s on to the next pause)
    if first is not None and quiet[:max(1, first - 25)].all():   # (the voice's own rise, up to 0.25 s, not counted)
        on = [first / 100] + on
    out = []
    for t in on:
        if not out or t - out[-1] > 0.2:
            out.append(t)
    return np.array(out)


def timed_lines(lines, x, words=None, korean=None):
    """The timing of a piece's lines (seconds into x). Measured against clear speech starts and two recognisers'
    word ends (2026-10-17/18):
      lyrics  -- kept as given (on-screen captions, or timed on the singer by song_check)
      start   -- where the voice comes back after a pause, nearest Gemini's start (0.8 s before .. 2 s after); else a
                 pause onset up to 2 s earlier the voice runs on from; else where the speech detector hears speech
                 begin; else the plain snap to sound. Over a music bed, moved to where the speech really begins.
      end     -- where Whisper hears the line's last word (words, with korean: English line -> its Korean); else the
                 speech detector's stop near Gemini's end; then never held 1 s+ past clearly heard speech.
    A spoken line of 2 s+ with no speech under it was invented over music and is dropped."""
    qon, qon_t = quiet_onsets(x), None
    offs = speech_ends(x)
    snapped = {l[2]: l for l in snap_to_speech_plain(lines, x)}
    talk = []
    if _vad is not None:
        try:
            talk = [(r['start'] / 16000, r['end'] / 16000) for r in
                    _vad(np.asarray(x, dtype=np.float32), _VadOptions(min_silence_duration_ms=150, speech_pad_ms=0, threshold=0.5))]
        except Exception:
            talk = []
    def tolerant():
        nonlocal qon_t
        if qon_t is None:
            qon_t = quiet_onsets(x, tolerant=True)
        return qon_t
    def on_speech_start(t):
        """A start on the speech detector's start t: the tolerant pause onset just before it, or t - 0.05 (it marks
        starts a little late)."""
        q = tolerant()
        q = q[(q >= t - 0.3) & (q <= t + 0.05)] if len(q) else q
        return float(q[-1]) if len(q) else t - 0.05
    out, guesses = [], []
    for st, et, en in lines:
        if en.startswith('♪'):
            out.append([st, et, en]); guesses.append(st); continue
        if et - st >= 2.0 and talk:
            # a spoken line of 2 s+ with (almost) no speech under it was invented over music (2026-10-17: "Guided by
            # the WPK" over an instrumental); short quiet drama lines are kept (3 of 255 real lines score that low)
            if sum(max(0.0, min(b, et + 0.3) - max(a, st - 0.3)) for a, b in talk) < 0.05 * (et - st):
                log('dropped a line with no speech under it: %r' % en[:60])
                continue
        guess = st
        floor = (out[-1][0] + 0.6) if out else -1.0      # never before the line before has had its own start
        # start: Gemini starts lines early (median 0.3 s, often 1-2 s), so the search reaches further after
        c = qon[(qon >= max(guess - START_BEFORE_S, floor)) & (qon <= guess + START_AFTER_S)] if len(qon) else qon
        if not len(c):
            # Gemini started late, inside the sentence (2026-10-17 news 3:45 and 13:06:13): a pause onset up to 2 s
            # earlier (or just after its start), if the voice runs on from it to Gemini's start without a break
            q = tolerant()
            back = [b for b in q[(q >= max(guess - 2.0, floor)) & (q < guess + 0.1)] if voiced_seconds(x, b, guess) >= 0.7 * (guess - b)]
            if back:
                c = np.array(back[-1:])
        if not len(c) and talk:
            # no pause onset near: before the plain snap (which lands on any sound, a knock or a door), where the
            # speech detector hears speech begin, if no speech runs at Gemini's start nor at the snap (2026-10-17
            # 4:04:28: "What have you done?" snapped onto a bang 1.4 s before the words)
            p = snapped.get(en, (st,))[0]
            if not any(a <= guess < b for a, b in talk) and not any(p - 0.1 <= a <= p + 0.4 or a <= p < b for a, b in talk):
                ss = [a for a, b in talk if max(guess - START_BEFORE_S, floor) <= a <= guess + START_AFTER_S]
                if ss:
                    c = np.array([on_speech_start(ss[0])])
        s2 = float(c[np.argmin(np.abs(c - guess))]) if len(c) else snapped.get(en, (st,))[0]
        if out and not len(c) and s2 <= out[-1][0]:
            # the plain snap knows no order: two lines on one onset, or this one before the line before (2026-10-17
            # 2:17:33: a 12 s line was cut to nothing). The onset goes to the line Gemini put nearer it; the other
            # keeps Gemini's own start.
            pg = guesses[-1]
            if abs(s2 - guess) < abs(out[-1][0] - pg) and pg < s2 - 0.5 and (len(out) < 2 or pg > out[-2][0] + 0.6):
                out[-1][0] = pg
            else:
                s2 = max(floor, min(guess, et - 0.5))
        if talk and voiced_seconds(x, s2 - 0.5, s2) > 0.3 and not any(a <= s2 + 0.1 and b > s2 for a, b in talk):
            # over a music bed the sound rises when the music does: the line was snapped there, 0.4-2 s before anyone
            # speaks (user, 2026-10-18). Nobody speaking at that start and speech within 2 s: it starts with the speech.
            later = [a for a, b in talk if s2 + 0.4 <= a <= s2 + 2.0]
            if later and later[0] < et - 0.5:
                s2 = on_speech_start(later[0])
        # end
        span = word_span((korean or {}).get(en), words, st - 4, et + 4) if words else None
        if span and span[1] is not None and span[1] > s2 + 0.3:
            e2 = span[1] + WORD_END_SHIFT
        else:
            # the speech detector's stop nearest Gemini's end, 0.1 s earlier (it stops a little late); none near:
            # the last stop before it, within 3 s and keeping at least half the line (not a breath mid-sentence)
            c = offs[(np.abs(offs - et) <= 1.0) & (offs > s2 + 0.3)] if len(offs) else offs
            if not len(c) and len(offs):
                c = offs[(offs > s2 + 0.3) & (offs <= et + 0.3) & (offs >= et - 3.0) & (offs - s2 >= 0.5 * (et - st))][-1:]
            e2 = (float(c[np.argmin(np.abs(c - et))]) if len(c) else et) - END_TRIM_S
        inside = [(a, b) for a, b in talk if a < e2 and b > s2]
        if inside and e2 - s2 > 1.0:
            # held too long (user, 2026-10-18): heard clearly, the talking stopped 1 s+ before the end with nothing
            # after -- the line ends with the talking. Lines it hardly hears (singing) keep their end.
            if sum(min(b, e2) - max(a, s2) for a, b in inside) >= 0.4 * (e2 - s2) and max(b for _, b in inside) < e2 - 1.0:
                e2 = max(max(b for _, b in inside) - END_TRIM_S, s2 + 0.8)
        out.append([s2, max(e2, s2 + 0.5), en]); guesses.append(guess)
    out.sort(key=lambda l: l[0])
    for i in range(len(out) - 1):                         # a line never runs on past the start of the next
        if not out[i][2].startswith('♪') and out[i][1] > out[i + 1][0] - 0.05:
            out[i][1] = max(out[i][0] + 0.5, out[i + 1][0] - 0.05)
    return [tuple(l) for l in out]


def spoken_not_sung(lines, x, korean=None):
    """A line Gemini marked as a song but that the Silero speech detector hears as speech is speech: someone talking
    over music (user, 2026-10-17). Tested: real lyrics are covered by detected speech ~0% of their time (48 lines,
    two music videos: none over 50%), spoken lines ~83% (86-99% over 50%), the mislabelled narration 64-78%."""
    if _vad is None or not any(l[2].startswith('♪') for l in lines):
        return lines
    try:
        ts = _vad(np.asarray(x, dtype=np.float32), _VadOptions(min_silence_duration_ms=150, speech_pad_ms=0, threshold=0.5))
    except Exception:
        return lines
    out = []
    for st, et, en in lines:
        if en.startswith('♪'):
            heard = sum(max(0.0, min(t['end'] / 16000, et) - max(t['start'] / 16000, st)) for t in ts)
            if heard > SUNG_MAX_SPEECH * max(0.1, et - st):
                plain = en.lstrip('♪ ').strip()
                if korean is not None and en in korean:
                    korean[plain] = korean[en]
                en = plain
        out.append((st, et, en))
    return out


def snap_to_speech(lines, x, words=None, korean=None):
    """The timing of a chunk's lines (seconds into x, 16 kHz): see timed_lines. words: Speech-to-Text words for x (or
    None), korean: English line -> its Korean (Gemini.korean). First, a line over (near) silence is dropped: Gemini
    sometimes invents one there ("This is Korean Central Television" over 3 s of silence)."""
    if not lines or len(x) < 16000:
        return lines
    # (a line starting where this piece's sound has ended -- the second row of an on-screen verse running into the
    # next piece -- cannot be judged here and is kept)
    lines = [l for l in lines if l[0] > len(x) / 16000 - 0.5 or voiced_seconds(x, l[0] - 0.3, l[1] + 0.3) >= 0.08]
    if not lines:
        return lines
    lines = spoken_not_sung(lines, x, korean)
    return timed_lines(lines, x, words, korean)


def snap_to_speech_plain(lines, x):
    """The ordinary snap: each line's start to the nearest onset of sound (refined to 10 ms), its end to where the
    sound stops. The fallback start in timed_lines when no pause is near."""
    if not lines or len(x) < 16000:
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
        while j > k - 26 and voiced(j - 1, fl): j -= 1
        if j <= k - 26:
            j = k                    # sound runs on further back (a music bed, the sentence before): not a voice start
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
        song = t.startswith('♪')
        # a copy of the same words (the same sound transcribed twice, at a batch edge) overlaps the line before; a
        # similar line after it is real speech (2026-10-17: the weather for the next city, a narrator restating a
        # point -- dropped within 8 s, they left 5-15 s of speech without subtitles)
        if kept and a - kept[-1][1] < (0.0 if song else 1.0) and repeats(t.lstrip('♪ '), [kept[-1][2].lstrip('♪ ')], 0.85) and len(content_words(t)) >= 2:
            if not song and b > kept[-1][1]:       # the copy's speech still needs a subtitle: the kept line spreads
                kept[-1] = (kept[-1][0], b, kept[-1][2])   # over it (2026-10-17 5:43: 8 s left empty)
            continue                                 # a song's refrain is sung again and shown again (2026-10-16 13:45)
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


STOPWORDS = set('this that these those with have from will they them their there what when where which while would '
                'could should about into your yours been were also very just than then here such some more most much '
                'only over like make made does done said says each'.split())


def content_words(t):
    """The words that carry meaning: longer than three letters, not a common function word ("How is this production
    guaranteed?" and "This production process is similar." share only "production")."""
    return {w for w in re.findall(r"[a-z0-9']+", t.lower()) if len(w) > 3 and w not in STOPWORDS}


def repeats(text, shown, share=0.6):
    """The second pass over a carried segment says again what was just shown ("activities, including cooking
    practice" after "...art activities, and cooking practice"): most of its words are already on screen."""
    words = content_words(text)
    if not words:
        return False
    seen = set().union(*(content_words(t) for t in shown)) if shown else set()
    return len(words & seen) / len(words) >= share


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
                    lines = fill_gaps(self.gemini, flac, audio, secs, deadline, expect, lines)      # speech left out
                    lines = fix_foreign(self.gemini.key, lines)                                     # garbled script
                    if time.time() < deadline - 30:
                        lines = song_check(self.gemini.key, audio, lines, deadline=deadline - 15)                       # 'songs' that are speech
                if lines:                                   # timed once, here
                    cues = make_cues(snap_to_speech(lines, audio, None, self.gemini.korean if self.gemini else None), voice_starts(audio, pauses=True), open_end=True)
                else:
                    cues = []
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
                if d['kind'] == 'sound' and not LIVE_SUBS:
                    self.finish(d['seq'], [])                   # live subtitles off: the English track stays, empty
                elif d['kind'] == 'sound':
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
EPG_LOCK = threading.Lock()
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
        # survive a restart (2026-10-17: a restart just after KCTV's sign-off lost tomorrow's guide): on air since,
        # last sound and a pending opening read are kept in the status file and taken back if still recent
        mine = st.get(key, {})
        if mine.get('on_since') and time.time() - mine.get('last_sound', 0) < EPG_RING_S:
            self.on_since, self.last_sound = mine['on_since'], mine.get('last_sound', 0)
            log(key, 'programme guide: on air since', pyongyang_hm(self.on_since), '(Pyongyang), carried over a restart')
        if mine.get('opening_due') and time.time() - mine['opening_due'] < EPG_RING_S:
            self.opening_due = mine['opening_due']
        self.saved_at = 0

    def status(self):
        try:
            return json.load(open(EPG_STATUS))
        except Exception:
            return {}

    def note(self, **kw):
        with EPG_LOCK:                                   # three channels share the file
            st = self.status()
            st.setdefault(self.key, {}).update(kw, at=time.strftime('%F %T'))
            try:
                tmp = EPG_STATUS + '.tmp.%s' % self.key
                json.dump(st, open(tmp, 'w'), ensure_ascii=False, indent=1); os.replace(tmp, EPG_STATUS)
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
            if now - self.saved_at >= 60:
                self.saved_at = now
                self.note(on_since=self.on_since, last_sound=now, opening_due=self.opening_due)
        else:
            if self.quiet_since is None:
                self.quiet_since = now
            if self.on_since is not None and now - self.quiet_since >= SIGNOFF_QUIET_S:
                if self.last_sound - self.on_since >= MIN_ON_AIR_S:
                    self.signoff_hm = pyongyang_hm(self.last_sound)
                    self.note(signoff=self.signoff_hm)
                    self.run('closing', self.last_sound)
                self.on_since = None
                self.note(on_since=None, last_sound=self.last_sound)
        if self.opening_due and now >= self.opening_due:
            self.opening_due = None
            self.note(opening_due=None)
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
    gemini = Gemini(config_key(conf))
    # Speech-to-Text is not used (user, 2026-10-17: too expensive); stt_words stays for a test or a one-off
    chans = [Channel(c, r2, groq, gemini) for c in chans_cfg]
    for c in chans:
        c.start()
    threading.Thread(target=r2.expiry_loop, daemon=True).start()
    threading.Thread(target=sweeper, daemon=True).start()
    threading.Thread(target=status_writer, args=(chans, groq, gemini), daemon=True).start()
    log('subtitles running for', ', '.join(c.cfg['key'] for c in chans), '(dry run)' if DRY else '')
    while True:
        time.sleep(3600)
