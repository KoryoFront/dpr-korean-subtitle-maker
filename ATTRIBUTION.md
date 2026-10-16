# Data sources

- **Samhung dictionaries** (DPRK Korean dictionary and Korean-English dictionary, by the Korea Computer Centre),
  in the SQLite form published by Digital NK: https://digitalnk.com/blog/2020/05/08/porting-north-korean-dictionaries-with-rust/
  and https://github.com/digitalprk/dicrs
- **Place names**: GeoNames (https://www.geonames.org), licensed CC BY 4.0. Converted to DPRK house style
  (no breves or apostrophes, initial ㄹ kept: Ryongnam, County / Mt. forms).
- Subtitles are made with Google Gemini (your own API key).
