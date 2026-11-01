//! Cross-codec regression suite.
//!
//! Every audio codec is exercised end to end: encode -> decode, sizing,
//! determinism and graceful handling of malformed input. The point is to fail
//! loudly if a dependency bump, a feature-gating change or a refactor silently
//! breaks any codec's bitstream or framing.
//!
//! Opus uses `std`, G.729 is behind the `g729` feature; the relevant tests are
//! conditionally compiled so the file also builds with codecs disabled.

// The convenience `encode`/`decode` and the factory helpers are `std`-only.
#![cfg(feature = "std")]

use audio_codec::g722::{G722Decoder, G722Encoder};
#[cfg(feature = "g729")]
use audio_codec::g729::{G729Decoder, G729Encoder};
use audio_codec::{
    CodecError, CodecType, Decoder, Encoder, create_decoder, create_encoder, pcma, pcmu,
    telephone_event::TelephoneEventDecoder, telephone_event::TelephoneEventEncoder,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A deterministic sine: `ms` milliseconds at `sample_rate`, peak `amp`.
fn sine(sample_rate: u32, ms: u32, freq: f64, amp: f64) -> Vec<i16> {
    let n = (sample_rate as f64 * ms as f64 / 1000.0).round() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (amp * (2.0 * std::f64::consts::PI * freq * t).sin()).round() as i16
        })
        .collect()
}

fn rms(s: &[i16]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    (s.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
}

fn encode_all<E: Encoder>(e: &mut E, pcm: &[i16]) -> Vec<u8> {
    let max = e.max_encode_bytes(pcm.len());
    let mut buf = vec![0u8; max.max(1)];
    let n = e.encode_into(pcm, &mut buf).expect("encode_into");
    buf.truncate(n);
    buf
}

fn decode_all<D: Decoder>(d: &mut D, data: &[u8]) -> Vec<i16> {
    let max = d.max_decode_samples(data.len());
    let mut buf = vec![0i16; max.max(1)];
    let n = d.decode_into(data, &mut buf).expect("decode_into");
    buf.truncate(n);
    buf
}

fn assert_energy_ratio(decoded: &[i16], reference: &[i16], lo: f64, hi: f64) {
    let reference_rms = rms(reference);
    let decoded_rms = rms(decoded);
    let ratio = decoded_rms / reference_rms;
    assert!(
        (lo..=hi).contains(&ratio),
        "decoded/reference RMS ratio {ratio:.3} outside [{lo}, {hi}] \
         (decoded {decoded_rms:.1}, reference {reference_rms:.1})",
    );
}

// ---------------------------------------------------------------------------
// G.711 (PCMU / PCMA)
// ---------------------------------------------------------------------------

#[test]
fn pcmu_roundtrip_preserves_signal() {
    let pcm = sine(8000, 200, 1000.0, 12000.0);
    let mut enc = pcmu::PcmuEncoder::new();
    let mut dec = pcmu::PcmuDecoder::new();

    let packet = encode_all(&mut enc, &pcm);
    assert_eq!(packet.len(), pcm.len(), "mu-law is 1 byte per sample");
    let out = decode_all(&mut dec, &packet);
    assert_eq!(out.len(), pcm.len());
    assert_energy_ratio(&out, &pcm, 0.90, 1.10);
}

#[test]
fn pcma_roundtrip_preserves_signal() {
    let pcm = sine(8000, 200, 1000.0, 12000.0);
    let mut enc = pcma::PcmaEncoder::new();
    let mut dec = pcma::PcmaDecoder::new();

    let packet = encode_all(&mut enc, &pcm);
    assert_eq!(packet.len(), pcm.len(), "A-law is 1 byte per sample");
    let out = decode_all(&mut dec, &packet);
    assert_eq!(out.len(), pcm.len());
    assert_energy_ratio(&out, &pcm, 0.90, 1.10);
}

#[test]
fn pcma_pcmu_silence_stays_silent() {
    let pcm = vec![0i16; 160];

    let mut mu_enc = pcmu::PcmuEncoder::new();
    let mu_dec = &mut pcmu::PcmuDecoder::new();
    let mu = decode_all(mu_dec, &encode_all(&mut mu_enc, &pcm));
    assert!(mu.iter().all(|&s| s.abs() <= 8), "mu-law silence drifted");

    let mut a_enc = pcma::PcmaEncoder::new();
    let a_dec = &mut pcma::PcmaDecoder::new();
    let a = decode_all(a_dec, &encode_all(&mut a_enc, &pcm));
    assert!(a.iter().all(|&s| s.abs() <= 8), "A-law silence drifted");
}

/// Exhaustive round-trip over every 16-bit sample: the quantisation error of
/// the lookup tables must stay within the G.711 tolerance. A table regression
/// (wrong segment/step) makes the error explode.
#[test]
fn g711_roundtrip_error_stays_bounded() {
    let pcm: Vec<i16> = (0..=u16::MAX).map(|v| v as i16).collect();

    let mu_enc = &mut pcmu::PcmuEncoder::new();
    let mu_src = pcm.clone();
    let mu_out = decode_all(&mut pcmu::PcmuDecoder::new(), &encode_all(mu_enc, &pcm));
    let mu_err = max_abs_err(&mu_src, &mu_out);
    println!("MULAW_MAX_ABS_ERR {mu_err}");

    let a_enc = &mut pcma::PcmaEncoder::new();
    let a_out = decode_all(&mut pcma::PcmaDecoder::new(), &encode_all(a_enc, &pcm));
    let a_err = max_abs_err(&pcm, &a_out);
    println!("ALAW_MAX_ABS_ERR {a_err}");

    // Current tables top out at 644 (mu-law) / 519 (A-law); 1024 leaves a
    // margin for a deliberate table tweak while catching gross regressions.
    assert!(mu_err <= 1024, "mu-law max abs error {mu_err} too large");
    assert!(a_err <= 1024, "A-law max abs error {a_err} too large");
}

/// The G.711 µ-law encoder must match the canonical ITU-T / Sun `g711.c`
/// algorithm for every 16-bit input.
///
/// Regression guard: an earlier version XOR'd with a partial mask
/// (`0xFF ^ (0xFF >> shift)`) instead of complementing the whole byte, emitting
/// non-standard codes for ~12% of inputs. Decoded by any standard G.711
/// decoder (media players, peers) that added ~6 dB of audible quantization
/// noise ("hiss") to recordings.
#[test]
fn pcmu_encoder_matches_canonical_g711() {
    /// Canonical linear -> µ-law (ITU-T G.711 / Sun g711.c).
    fn canonical_ulaw(mut sample: i16) -> u8 {
        const BIAS: i32 = 0x84;
        const CLIP: i16 = 32635;
        if sample > CLIP {
            sample = CLIP;
        } else if sample < -CLIP {
            sample = -CLIP;
        }
        let sign: u8 = if sample < 0 { 0x80 } else { 0x00 };
        if sample < 0 {
            sample = -sample;
        }
        let s = sample as i32 + BIAS;
        // exp_lut[(s >> 7) & 0xFF] == floor(log2(x)) for x >= 2, else 0.
        let x = ((s >> 7) & 0xFF) as u32;
        let exponent: u8 = if x < 2 {
            0
        } else {
            (31 - x.leading_zeros()) as u8
        };
        let mantissa = ((s >> (exponent as i32 + 3)) & 0x0F) as u8;
        !(sign | (exponent << 4) | mantissa)
    }

    let mut enc = pcmu::PcmuEncoder::new();
    let mut out = [0u8; 1];
    let mut mismatches = 0u32;
    for i in 0..=u16::MAX {
        let sample = i as i16;
        let n = enc.encode_into(&[sample], &mut out).expect("encode_into");
        assert_eq!(n, 1);
        if out[0] != canonical_ulaw(sample) {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "µ-law mismatch: sample={sample} got={:#04x} want={:#04x}",
                    out[0],
                    canonical_ulaw(sample)
                );
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches}/65536 µ-law codes differ from canonical G.711"
    );
}

/// Issue #5 spec check: for every 16-bit input, encode -> decode must stay
/// within **half a quantisation step** of the input.
///
/// The step size of the µ-law segment holding the input is `8 << segment`
/// (`linear2ulaw_algo` clips to ±CLIP first, so the segment comes from the
/// clipped value). Half a step therefore bounds the error of a correct
/// encoder/decoder pair; the buggy pre-0.4.11 encoder mirrored codes inside
/// segments 0-4 and missed this bound for ~93% of inputs.
#[test]
fn pcmu_roundtrip_within_half_quantization_step() {
    const BIAS: i32 = 0x84;
    const CLIP: i16 = 32635;

    // Known-answer vectors from the issue report (standard G.711 column).
    // The pre-0.4.11 encoder produced 0xE7 / 0xEC / 0xCD / 0xB6 / 0xFE here.
    let mut enc = pcmu::PcmuEncoder::new();
    let mut out = [0u8; 1];
    for (sample, want) in [
        (0i16, 0xFFu8),
        (240, 0xE8),
        (330, 0xE3),
        (1000, 0xCE),
        (3000, 0xB7),
    ] {
        enc.encode_into(&[sample], &mut out).expect("encode_into");
        assert_eq!(out[0], want, "µ-law code for {sample}");
    }

    let mut dec = pcmu::PcmuDecoder::new();
    let mut pcm = [0i16; 1];
    let mut worst: (i32, i16) = (0, 0);
    for i in 0..=u16::MAX {
        let sample = i as i16;
        let n = enc.encode_into(&[sample], &mut out).expect("encode_into");
        assert_eq!(n, 1);
        let n = dec.decode_into(&out, &mut pcm).expect("decode_into");
        assert_eq!(n, 1);

        // Segment of the (clipped) input: same loop as `linear2ulaw_algo`.
        // |sample| beyond ±CLIP only loses the clip distance, so the bound
        // applies to the clipped value.
        let clipped = sample.clamp(-CLIP, CLIP);
        let mut seg: u32 = 0;
        let mut v = (clipped as i32).abs() + BIAS;
        while v >= 256 {
            seg += 1;
            v >>= 1;
        }
        let half_step = 4i32 << seg;
        let err = (pcm[0] as i32 - clipped as i32).abs();
        assert!(
            err <= half_step,
            "µ-law round trip off by {err} (half step {half_step}) \
             for sample {sample} -> {:#04x}",
            out[0]
        );
        if err > worst.0 {
            worst = (err, sample);
        }
    }
    println!("PCMU half-step worst error: {worst:?}");
}

fn max_abs_err(a: &[i16], b: &[i16]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).abs())
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// G.722
// ---------------------------------------------------------------------------

#[test]
fn g722_roundtrip_preserves_signal() {
    let pcm = sine(16000, 200, 1000.0, 12000.0);
    let mut enc = G722Encoder::new();
    let mut dec = G722Decoder::new();

    let packet = encode_all(&mut enc, &pcm);
    assert_eq!(packet.len(), pcm.len() / 2, "G.722 packs 2 samples/byte");
    let out = decode_all(&mut dec, &packet);
    assert_eq!(out.len(), pcm.len());
    assert_energy_ratio(&out, &pcm, 0.80, 1.20);
}

#[test]
fn g722_encode_is_deterministic() {
    let pcm = sine(16000, 40, 1000.0, 12000.0);
    let a = encode_all(&mut G722Encoder::new(), &pcm);
    let b = encode_all(&mut G722Encoder::new(), &pcm);
    assert_eq!(a, b, "G.722 encoder is not deterministic");
}

// ---------------------------------------------------------------------------
// G.729
// ---------------------------------------------------------------------------

#[cfg(feature = "g729")]
#[test]
fn g729_roundtrip_is_sized_and_deterministic() {
    let pcm = sine(8000, 100, 500.0, 10000.0); // 800 samples = 10 frames
    let mut enc = G729Encoder::new();
    let a = encode_all(&mut enc, &pcm);

    let mut enc2 = G729Encoder::new();
    let b = encode_all(&mut enc2, &pcm);
    assert_eq!(a, b, "G.729 encoder is not deterministic");

    assert_eq!(a.len(), pcm.len() / 8, "G.729: 80 samples -> 10 bytes");
    assert_eq!(a.len(), 100);

    let mut dec = G729Decoder::new();
    let out = decode_all(&mut dec, &a);
    assert_eq!(out.len(), pcm.len());
    assert_energy_ratio(&out, &pcm, 0.50, 1.50);
}

#[cfg(feature = "g729")]
#[test]
fn g729_truncated_input_does_not_panic() {
    let mut dec = G729Decoder::new();

    // Shorter than a single 10-byte frame: nothing to decode, no panic.
    assert!(decode_all(&mut dec, &[0u8; 5]).is_empty());
    assert!(decode_all(&mut dec, &[]).is_empty());

    // One full frame plus a 7-byte tail: exactly one frame is consumed.
    let out = decode_all(&mut dec, &[0u8; 17]);
    assert_eq!(out.len(), 80);
}

/// Locks the G.729 encoder output for a deterministic frame so a dependency
/// bump (e.g. `g729-sys` 0.2.0 -> 0.2.1) that changes the bitstream fails loud.
#[cfg(feature = "g729")]
#[test]
fn g729_encode_is_bit_exact() {
    let pcm: Vec<i16> = (0..80)
        .map(|i| ((((i * 97) % 512) - 256) * 100) as i16)
        .collect();
    let mut enc = G729Encoder::new();
    let packet = encode_all(&mut enc, &pcm);
    let hex: String = packet.iter().map(|b| format!("{b:02x}")).collect();
    println!("G729_HEX n={} {hex}", packet.len());
    assert_eq!(packet.len(), 10);
    assert_eq!(hex, "f8b1cb0000fada3f0dac");
}

// ---------------------------------------------------------------------------
// Opus
// ---------------------------------------------------------------------------

#[cfg(feature = "opus")]
mod opus {
    use super::*;
    use audio_codec::opus::{OpusDecoder, OpusEncoder};

    #[test]
    fn opus_mono_roundtrip_preserves_signal() {
        let pcm = sine(48000, 20, 440.0, 12000.0); // 960 mono samples
        let mut enc = OpusEncoder::new_default();
        let packet = enc.encode(&pcm);
        assert!(!packet.is_empty(), "Opus encoder produced no output");

        let mut dec = OpusDecoder::new_default();
        let out = dec.decode(&packet);
        assert_eq!(out.len(), 960, "20ms @ 48kHz should decode to 960 mono");
        assert_energy_ratio(&out, &pcm, 0.50, 1.50);
    }

    #[test]
    fn opus_mono_encoder_roundtrips_mono() {
        let pcm = sine(48000, 20, 440.0, 12000.0);
        let mut enc = OpusEncoder::new(48000, 1);
        let packet = encode_all(&mut enc, &pcm);
        assert!(!packet.is_empty());

        let mut dec = OpusDecoder::new(48000, 1);
        let out = decode_all(&mut dec, &packet);
        assert_eq!(out.len(), 960);
        assert_energy_ratio(&out, &pcm, 0.50, 1.50);
    }

    #[test]
    fn opus_consecutive_frames_keep_frame_size() {
        let pcm = sine(48000, 20, 440.0, 12000.0);
        let mut enc = OpusEncoder::new_default();
        let packet = enc.encode(&pcm);

        let mut dec = OpusDecoder::new_default();
        for frame in 0..5 {
            let out = dec.decode(&packet);
            assert_eq!(out.len(), 960, "frame {frame} changed length");
        }
    }

    #[test]
    fn opus_empty_input_is_empty() {
        let mut enc = OpusEncoder::new_default();
        assert!(enc.encode(&[]).is_empty());
        let mut dec = OpusDecoder::new_default();
        assert!(dec.decode(&[]).is_empty());
    }
}

// ---------------------------------------------------------------------------
// Telephone Event (RFC 4733): carries no audio
// ---------------------------------------------------------------------------

#[test]
fn telephone_event_is_a_noop_codec() {
    let mut enc = TelephoneEventEncoder::new();
    assert_eq!(enc.max_encode_bytes(160), 0);
    let mut buf = [0u8; 4];
    assert_eq!(enc.encode_into(&[0i16; 160], &mut buf).unwrap(), 0);

    let mut dec = TelephoneEventDecoder::new();
    assert_eq!(dec.max_decode_samples(4), 0);
    let mut out = [0i16; 4];
    assert_eq!(dec.decode_into(&[0u8; 4], &mut out).unwrap(), 0);
}

// ---------------------------------------------------------------------------
// CodecType metadata / parsing
// ---------------------------------------------------------------------------

#[test]
fn codec_metadata_is_stable() {
    assert_eq!(CodecType::PCMU.payload_type(), 0);
    assert_eq!(CodecType::PCMA.payload_type(), 8);
    assert_eq!(CodecType::G722.payload_type(), 9);
    assert_eq!(CodecType::TelephoneEvent.payload_type(), 101);
    assert_eq!(CodecType::PCMU.mime_type(), "audio/PCMU");
    assert_eq!(CodecType::PCMA.rtpmap(), "PCMA/8000");
    assert_eq!(CodecType::G722.rtpmap(), "G722/8000");
    assert_eq!(CodecType::G722.samplerate(), 16000);
    assert_eq!(CodecType::PCMU.clock_rate(), 8000);
    assert!(CodecType::PCMA.is_audio());
    assert!(!CodecType::TelephoneEvent.is_audio());
    assert!(CodecType::TelephoneEvent.is_dynamic());
    assert_eq!(CodecType::try_from("ulaw").unwrap(), CodecType::PCMU);
    assert_eq!(CodecType::try_from("ALAW").unwrap(), CodecType::PCMA);
}

#[cfg(feature = "g729")]
#[test]
fn g729_metadata_and_parsing() {
    assert_eq!(CodecType::G729.payload_type(), 18);
    assert_eq!(CodecType::G729.mime_type(), "audio/G729");
    assert_eq!(CodecType::G729.rtpmap(), "G729/8000");
    assert_eq!(CodecType::try_from(18u8).unwrap(), CodecType::G729);
    assert_eq!(CodecType::try_from("g729").unwrap(), CodecType::G729);
    assert_eq!(CodecType::try_from("G729").unwrap(), CodecType::G729);
}

#[cfg(feature = "opus")]
#[test]
fn opus_metadata_and_parsing() {
    assert_eq!(CodecType::Opus.payload_type(), 111);
    assert_eq!(CodecType::Opus.mime_type(), "audio/opus");
    assert_eq!(CodecType::Opus.rtpmap(), "opus/48000/2");
    assert_eq!(CodecType::try_from(111u8).unwrap(), CodecType::Opus);
    assert_eq!(CodecType::try_from("opus").unwrap(), CodecType::Opus);
}

// ---------------------------------------------------------------------------
// Factory API used by callers (`create_encoder` / `create_decoder`)
// ---------------------------------------------------------------------------

#[test]
fn factory_roundtrips_pcmu_pcma() {
    for codec in [CodecType::PCMU, CodecType::PCMA] {
        let mut enc = create_encoder(codec);
        let mut dec = create_decoder(codec);
        let pcm = sine(8000, 20, 1000.0, 12000.0); // 160 samples
        let data = enc.encode(&pcm);
        assert_eq!(data.len(), 160, "{codec:?}");
        let out = dec.decode(&data);
        assert_eq!(out.len(), 160, "{codec:?}");
    }
}

#[test]
fn factory_roundtrips_g722() {
    let mut enc = create_encoder(CodecType::G722);
    let mut dec = create_decoder(CodecType::G722);
    let pcm = sine(16000, 20, 1000.0, 12000.0); // 320 samples
    let data = enc.encode(&pcm);
    assert_eq!(data.len(), 160);
    let out = dec.decode(&data);
    assert_eq!(out.len(), 320);
}

#[cfg(feature = "g729")]
#[test]
fn factory_roundtrips_g729() {
    let mut enc = create_encoder(CodecType::G729);
    let mut dec = create_decoder(CodecType::G729);
    let pcm = sine(8000, 20, 500.0, 10000.0); // 160 samples = 2 frames
    let data = enc.encode(&pcm);
    assert_eq!(data.len(), 20);
    let out = dec.decode(&data);
    assert_eq!(out.len(), 160);
}

#[cfg(feature = "opus")]
#[test]
fn factory_roundtrips_opus() {
    let mut enc = create_encoder(CodecType::Opus);
    let mut dec = create_decoder(CodecType::Opus);
    let pcm = sine(48000, 20, 440.0, 12000.0); // 960 mono
    let data = enc.encode(&pcm);
    assert!(!data.is_empty());
    let out = dec.decode(&data);
    assert_eq!(out.len(), 960);
}

// ---------------------------------------------------------------------------
// Buffer-size contract
// ---------------------------------------------------------------------------

#[test]
fn undersized_buffers_report_buffer_too_small() {
    let mut enc = pcmu::PcmuEncoder::new();
    let err = enc.encode_into(&[0i16; 16], &mut [0u8; 8]).unwrap_err();
    assert_eq!(err, CodecError::BufferTooSmall);

    let mut dec = pcmu::PcmuDecoder::new();
    let err = dec.decode_into(&[0u8; 16], &mut [0i16; 8]).unwrap_err();
    assert_eq!(err, CodecError::BufferTooSmall);
}
