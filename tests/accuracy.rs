//! Accuracy verification against "real" musical content.
//!
//! A short MIDI-style melody (harmonic additive synthesis + note envelopes,
//! plus a bass line for the polyphonic variants) is synthesised at each
//! codec's native rate, encoded/decoded, then scored against the original with
//! delay-aligned SNR (gain-compensated), Pearson correlation and RMS level.
//! A second set of tests pushes the signal *through the resampler* as well and
//! checks that the RMS stays close to the original.
//!
//! Pure-tone tests can pass while a codec mangles music; these thresholds give
//! a per-codec quality floor and print the measured numbers so a regression is
//! visible even before it trips an assertion.
//!
//! G.729 is an 8 kbit/s *speech* codec, so it is scored on a monophonic melody
//! (its intended content); Opus is scored in `Audio` (music) mode.

// The convenience `encode`/`decode` and the factory helpers are `std`-only.
#![cfg(feature = "std")]

use audio_codec::g722::{G722Decoder, G722Encoder};
#[cfg(feature = "g729")]
use audio_codec::g729::{G729Decoder, G729Encoder};
use audio_codec::pcma::{PcmaDecoder, PcmaEncoder};
use audio_codec::pcmu::{PcmuDecoder, PcmuEncoder};
use audio_codec::{Decoder, Encoder, resample};

const A4_HZ: f64 = 440.0;
const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// MIDI-style music synthesis
// ---------------------------------------------------------------------------

/// A note in the score: MIDI number, start time and duration in seconds.
struct Note {
    midi: f64,
    start: f64,
    dur: f64,
    amp: f64,
}

fn midi_to_freq(note: f64) -> f64 {
    A4_HZ * 2f64.powf((note - 69.0) / 12.0)
}

/// Additive-synthesis note: 6 harmonics, 5 ms attack, exponential decay and a
/// gentle vibrato so the signal has realistic spectral/temporal complexity.
fn render_note(out: &mut [f64], sr: f64, n: &Note) {
    let f0 = midi_to_freq(n.midi);
    let harmonics = [1.0, 0.5, 0.33, 0.2, 0.12, 0.07];
    let start = (n.start * sr) as usize;
    let len = (n.dur * sr) as usize;

    for i in 0..len {
        let idx = start + i;
        if idx >= out.len() {
            break;
        }
        let t = i as f64 / sr;
        let attack = ((i as f64) / (0.005 * sr)).min(1.0);
        let env = attack * (-3.0 * t).exp();
        let vib = 1.0 + 0.004 * (2.0 * PI * 5.0 * t).sin();

        let mut s = 0.0;
        for (h, &a) in harmonics.iter().enumerate() {
            s += a * (2.0 * PI * f0 * (h as f64 + 1.0) * vib * t).sin();
        }
        out[idx] += n.amp * env * s;
    }
}

/// Ode to Joy (first two phrases) as (MIDI note, beats); 0.25 s per beat.
const MELODY: &[(f64, f64)] = &[
    (64., 1.),
    (64., 1.),
    (65., 1.),
    (67., 1.),
    (67., 1.),
    (65., 1.),
    (64., 1.),
    (62., 1.),
    (60., 1.),
    (60., 1.),
    (62., 1.),
    (64., 1.),
    (64., 1.5),
    (62., 0.5),
    (62., 2.),
];

const BASS: &[(f64, f64)] = &[
    (48., 2.),
    (48., 2.),
    (43., 2.),
    (43., 2.),
    (41., 2.),
    (41., 2.),
    (48., 2.),
    (48., 2.),
];

const BEAT: f64 = 0.25;

fn build(sr: u32, with_bass: bool) -> Vec<i16> {
    let melody_amp = if with_bass { 0.55 } else { 0.7 };
    let mut notes = Vec::new();
    let mut t = 0.0;
    for &(midi, beats) in MELODY {
        let dur = beats * BEAT;
        notes.push(Note {
            midi,
            start: t,
            dur,
            amp: melody_amp,
        });
        t += dur;
    }
    let total = t;
    if with_bass {
        for (i, &(midi, beats)) in BASS.iter().enumerate() {
            notes.push(Note {
                midi,
                start: i as f64 * 2.0 * BEAT,
                dur: beats * BEAT,
                amp: 0.5,
            });
        }
    }

    let n = (sr as f64 * total) as usize;
    let mut buf = vec![0.0f64; n];
    for note in &notes {
        render_note(&mut buf, sr as f64, note);
    }

    let peak = buf.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let scale = 0.9 * 32767.0 / peak.max(1e-9);
    buf.iter()
        .map(|&v| (v * scale).clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// Polyphonic music (melody + bass).
fn music(sr: u32) -> Vec<i16> {
    build(sr, true)
}

/// Monophonic melody (speech-like, for G.729).
#[cfg(feature = "g729")]
fn mono_music(sr: u32) -> Vec<i16> {
    build(sr, false)
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

fn rms(s: &[i16]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    (s.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
}

/// Pearson correlation of `x[i]` with `y[i + shift]`, sampled every `stride`.
fn correlation_at(x: &[i16], y: &[i16], shift: i32, stride: usize) -> Option<f64> {
    let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    let xs = x.len() as i64;
    let ys = y.len() as i64;
    let i0 = 0.max(-(shift as i64));
    let i1 = xs.min(ys - shift as i64);
    let mut i = i0;
    while i < i1 {
        let a = x[i as usize] as f64;
        let b = y[(i + shift as i64) as usize] as f64;
        n += 1.0;
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
        i += stride as i64;
    }
    if n < 8.0 {
        return None;
    }
    let cov = sxy / n - (sx / n) * (sy / n);
    let vx = sxx / n - (sx / n).powi(2);
    let vy = syy / n - (sy / n).powi(2);
    if vx <= 0.0 || vy <= 0.0 {
        return None;
    }
    Some(cov / (vx.sqrt() * vy.sqrt()))
}

/// Best delay alignment by coarse-to-fine correlation search.
fn best_shift(x: &[i16], y: &[i16], max_lag: i32) -> i32 {
    let mut coarse = 0i32;
    let mut best_r = f64::MIN;
    let mut s = -max_lag;
    while s <= max_lag {
        if let Some(r) = correlation_at(x, y, s, 8)
            && r > best_r
        {
            best_r = r;
            coarse = s;
        }
        s += 4;
    }
    let mut fine = coarse;
    let mut fine_r = f64::MIN;
    for s in (coarse - 4)..=(coarse + 4) {
        if let Some(r) = correlation_at(x, y, s, 1)
            && r > fine_r
        {
            fine_r = r;
            fine = s;
        }
    }
    fine
}

/// Gain-compensated SNR (dB) and the least-squares gain, over the overlap.
fn snr_at(x: &[i16], y: &[i16], shift: i32) -> (f64, f64) {
    let (mut sxx, mut syy, mut sxy) = (0f64, 0.0, 0.0);
    let xs = x.len() as i64;
    let ys = y.len() as i64;
    let i0 = 0.max(-(shift as i64));
    let i1 = xs.min(ys - shift as i64);
    for i in i0..i1 {
        let a = x[i as usize] as f64;
        let b = y[(i + shift as i64) as usize] as f64;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    if syy <= 0.0 {
        return (f64::NEG_INFINITY, 0.0);
    }
    let gain = sxy / syy;
    let err = (sxx - 2.0 * gain * sxy + gain * gain * syy).max(0.0);
    if err <= 0.0 || sxx <= 0.0 {
        return (f64::INFINITY, gain);
    }
    (10.0 * (sxx / err).log10(), gain)
}

/// Score a codec: align, then check length, correlation, SNR and RMS level.
fn assert_accuracy(
    name: &str,
    original: &[i16],
    decoded: &[i16],
    max_lag: i32,
    min_snr: f64,
    min_corr: f64,
    rms_band: (f64, f64),
) {
    assert_eq!(
        decoded.len(),
        original.len(),
        "{name}: decoded length {} != input {}",
        decoded.len(),
        original.len()
    );
    let shift = best_shift(original, decoded, max_lag);
    let corr = correlation_at(original, decoded, shift, 1).unwrap_or(0.0);
    let (snr, gain) = snr_at(original, decoded, shift);
    let rms_ratio = rms(decoded) / rms(original);
    println!(
        "ACCURACY {name}: shift={shift} corr={corr:.4} snr={snr:.1}dB gain={gain:.3} rms={rms_ratio:.3}"
    );
    assert!(
        corr >= min_corr,
        "{name}: correlation {corr:.4} < {min_corr}"
    );
    assert!(snr >= min_snr, "{name}: SNR {snr:.1}dB < {min_snr}dB");
    assert!(
        (rms_band.0..=rms_band.1).contains(&rms_ratio),
        "{name}: RMS ratio {rms_ratio:.3} outside {rms_band:?}"
    );
}

// ---------------------------------------------------------------------------
// Encode/decode helpers
// ---------------------------------------------------------------------------

fn encode_all<E: Encoder>(e: &mut E, pcm: &[i16]) -> Vec<u8> {
    let mut buf = vec![0u8; e.max_encode_bytes(pcm.len()).max(1)];
    let n = e.encode_into(pcm, &mut buf).unwrap();
    buf.truncate(n);
    buf
}

fn decode_all<D: Decoder>(d: &mut D, data: &[u8]) -> Vec<i16> {
    let mut buf = vec![0i16; d.max_decode_samples(data.len()).max(1)];
    let n = d.decode_into(data, &mut buf).unwrap();
    buf.truncate(n);
    buf
}

// ---------------------------------------------------------------------------
// Per-codec accuracy (native rate, no resampler)
// ---------------------------------------------------------------------------

#[test]
fn pcmu_accuracy_on_music() {
    let pcm = music(8000);
    let out = decode_all(
        &mut PcmuDecoder::new(),
        &encode_all(&mut PcmuEncoder::new(), &pcm),
    );
    assert_accuracy("PCMU", &pcm, &out, 2, 30.0, 0.995, (0.9, 1.1));
}

#[test]
fn pcma_accuracy_on_music() {
    let pcm = music(8000);
    let out = decode_all(
        &mut PcmaDecoder::new(),
        &encode_all(&mut PcmaEncoder::new(), &pcm),
    );
    assert_accuracy("PCMA", &pcm, &out, 2, 30.0, 0.995, (0.9, 1.1));
}

#[test]
fn g722_accuracy_on_music() {
    let pcm = music(16000);
    let out = decode_all(
        &mut G722Decoder::new(),
        &encode_all(&mut G722Encoder::new(), &pcm),
    );
    assert_accuracy("G722", &pcm, &out, 64, 30.0, 0.99, (0.9, 1.1));
}

#[cfg(feature = "g729")]
#[test]
fn g729_accuracy_on_music() {
    // G.729 is a speech codec: score it on a monophonic melody.
    let pcm = mono_music(8000);
    let out = decode_all(
        &mut G729Decoder::new(),
        &encode_all(&mut G729Encoder::new(), &pcm),
    );
    assert_accuracy("G729", &pcm, &out, 300, 4.0, 0.80, (0.7, 1.2));
}

#[cfg(feature = "opus")]
#[test]
fn opus_accuracy_on_music() {
    use audio_codec::opus::{OpusApplication, OpusDecoder, OpusEncoder};

    let pcm = music(48000);

    // Music content: use the `Audio` application, not the speech-oriented
    // default. Packet based: encode/decode 20 ms (960-sample) frames.
    let mut enc = OpusEncoder::new_with_application(48000, 1, OpusApplication::Audio);
    enc.set_bitrate(64_000);
    let mut dec = OpusDecoder::new(48000, 1);
    let mut out = Vec::with_capacity(pcm.len());
    for frame in pcm.chunks(960) {
        let packet = encode_all(&mut enc, frame);
        out.extend_from_slice(&decode_all(&mut dec, &packet));
    }
    // Drop the codec's look-ahead tail so lengths line up.
    out.truncate(pcm.len());
    assert_accuracy("Opus", &pcm, &out, 1000, 15.0, 0.98, (0.9, 1.1));
}

// ---------------------------------------------------------------------------
// Resampler-inclusive pipelines
// ---------------------------------------------------------------------------

/// Check that the processed RMS is close to the original RMS (level accuracy).
fn assert_rms_close(name: &str, original: &[i16], processed: &[i16], lo: f64, hi: f64) {
    let n = original.len().min(processed.len());
    let o = rms(&original[..n]);
    let p = rms(&processed[..n]);
    let ratio = p / o;
    println!("RMS {name}: original={o:.1} processed={p:.1} ratio={ratio:.4}");
    assert!(
        (lo..=hi).contains(&ratio),
        "{name}: RMS ratio {ratio:.3} outside [{lo}, {hi}]"
    );
}

#[test]
fn resampler_roundtrip_preserves_rms() {
    let base = music(48000);
    for rate in [8000u32, 16000, 24000, 32000] {
        let down = resample(&base, 48000, rate);
        let up = resample(&down, rate, 48000);
        assert_rms_close(&format!("48k->{rate}->48k"), &base, &up, 0.95, 1.05);
    }
}

#[test]
fn g711_through_resampler_preserves_rms() {
    let base = music(48000);
    let at8k = resample(&base, 48000, 8000);

    let mu = decode_all(
        &mut PcmuDecoder::new(),
        &encode_all(&mut PcmuEncoder::new(), &at8k),
    );
    assert_rms_close(
        "PCMU via resampler",
        &base,
        &resample(&mu, 8000, 48000),
        0.9,
        1.1,
    );

    let a = decode_all(
        &mut PcmaDecoder::new(),
        &encode_all(&mut PcmaEncoder::new(), &at8k),
    );
    assert_rms_close(
        "PCMA via resampler",
        &base,
        &resample(&a, 8000, 48000),
        0.9,
        1.1,
    );
}

#[cfg(feature = "g729")]
#[test]
fn g729_through_resampler_preserves_rms() {
    let base = music(48000);
    let at8k = resample(&base, 48000, 8000);
    let out = decode_all(
        &mut G729Decoder::new(),
        &encode_all(&mut G729Encoder::new(), &at8k),
    );
    assert_rms_close(
        "G729 via resampler",
        &base,
        &resample(&out, 8000, 48000),
        0.5,
        1.6,
    );
}

#[test]
fn g722_through_resampler_preserves_rms() {
    let base = music(48000);
    let at16k = resample(&base, 48000, 16000);
    let out = decode_all(
        &mut G722Decoder::new(),
        &encode_all(&mut G722Encoder::new(), &at16k),
    );
    assert_rms_close(
        "G722 via resampler",
        &base,
        &resample(&out, 16000, 48000),
        0.7,
        1.3,
    );
}
