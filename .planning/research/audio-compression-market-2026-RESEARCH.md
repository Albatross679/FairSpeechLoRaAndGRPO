# Audio Compression Market Traditions for ASR Evaluation

Researched: 2026-04-29

Confidence: medium-high. Current product limits can change, so consumer-service rows should be rechecked before final paper submission.

## Why This Matters

FairSpeech full inference uses one clean 16 kHz PCM baseline plus five compressed or bandwidth-limited variants. The chosen variants map to common speech-audio stress regimes: telephony/narrowband, medium-band web speech, low-bitrate podcast/web delivery, and severe lossy compression.

## Market Snapshot

| Market setting | Typical codecs | Common sample rates | Typical bitrate range | FairSpeech analogue |
|---|---:|---:|---:|---|
| ASR / lab reference | PCM WAV, FLAC | 16 kHz for speech; 44.1/48 kHz for media | 16 kHz mono PCM is 256 kbps | `baseline` |
| Streaming music | AAC, Ogg Vorbis, FLAC | 44.1/48 kHz | 24-320 kbps lossy; lossless tier higher | `mp3_64k` upper lossy stress |
| Podcast hosting | MP3, AAC, WAV, FLAC | 22.05/24/44.1/48 kHz | 40-128 kbps common speech range | `mp3_64k`, `mp3_32k` |
| Web audio | MP3, AAC, Opus, Vorbis, FLAC | 16-48 kHz depending codec | MP3 can span 8-320 kbps | `mp3_16k` to `mp3_64k` |
| Legacy telephony | G.711, AMR-NB | 8 kHz | 4.75-64 kbps | `bottleneck_8k` |
| Wideband mobile speech | AMR-WB, EVS | 16 kHz and wider | 6.6-128 kbps | `bottleneck_12k`, `mp3_32k` |
| Real-time web calls | Opus | 8/12/16/24/48 kHz effective | 6-510 kbps; speech often much lower | `bottleneck_12k`, `mp3_32k` |
| Bluetooth audio | LC3, LDAC and vendor codecs | 8-48 kHz | 16-320 kbps for LC3; higher for LDAC | mostly out-of-scope for current matrix |

## ASR Implications

| Concept | Meaning | Why it affects ASR |
|---|---|---|
| Bitrate | Encoded bits per second after compression. | Lower bitrate can smear consonants, remove quiet speech cues, and create codec artifacts. |
| Resampling rate | Number of waveform samples per second. | Lower sample rates remove high-frequency speech information before the ASR model sees it. |
| Codec artifacts | Distortion introduced by lossy encoders. | Models trained on clean 16 kHz speech may treat artifacts as noise or hallucination triggers. |
| Model input handling | Most ASR frontends normalize audio into model-specific features. | Wav2Vec2 commonly consumes 16 kHz waveforms; Whisper resamples to 16 kHz log-Mel features; multimodal Gen3 ASR wrappers vary by processor. |

## Sources

| ID | Source | Notes |
|---|---|---|
| S1 | [Spotify audio quality](https://support.spotify.com/us/article/audio-quality/) | Consumer streaming bitrate tiers. |
| S2 | [Apple Podcasts audio requirements](https://podcasters.apple.com/support/893-audio-requirements) | Podcast codec, sample-rate, and bitrate guidance. |
| S3 | [Apple Music lossless support](https://support.apple.com/en-us/118295) | Consumer lossless sample-rate ranges. |
| S4 | [MDN audio codec guide](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Audio_codecs) | Web codec formats and MP3 operating ranges. |
| S5 | [MDN WebRTC codec guide](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/WebRTC_codecs) | WebRTC codec defaults and Opus context. |
| S6 | [RFC 6716: Opus](https://datatracker.ietf.org/doc/html/rfc6716) | Opus bitrate and supported bandwidth modes. |
| S7 | [RFC 7874: WebRTC audio](https://datatracker.ietf.org/doc/html/rfc7874) | WebRTC audio interoperability requirements. |
| S8 | [ITU G.711](https://www.itu.int/rec/t-rec-g.711-198811-i/en) | Classic 8 kHz, 64 kbps telephony baseline. |
| S9 | [3GPP EVS overview](https://www.3gpp.org/news-events/3gpp-news/evs-news) | EVS codec bandwidth and rate family. |
| S10 | [Bluetooth LE Audio specifications](https://www.bluetooth.com/learn-about-bluetooth/feature-enhancements/le-audio/le-audio-specifications/) | LC3 sample-rate and bitrate envelope. |
