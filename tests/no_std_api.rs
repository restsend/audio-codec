//! Integration tests for the slice-based `*_into` API (no_std-compatible).
//!
//! These tests exercise the same code paths that bare-metal users hit when
//! they call `encode_into`/`decode_into`/`resample_into` directly.

use audio_codec::{
    CodecError, Decoder, Encoder, bytes_to_samples_into, g722, g729, pcma, pcmu,
    samples_to_bytes_into,
};

/// Like `roundtrip` but tolerates quantization error (for lossy codecs).
fn roundtrip_approx<E: Encoder, D: Decoder>(
    encoder: &mut E,
    decoder: &mut D,
    pcm: &[i16],
    tol: i32,
) {
    let max_bytes = encoder.max_encode_bytes(pcm.len());
    let mut enc_buf = vec![0u8; max_bytes];
    let n = encoder.encode_into(pcm, &mut enc_buf).expect("encode_into");
    let enc_buf = &enc_buf[..n];

    let max_samples = decoder.max_decode_samples(n);
    let mut dec_buf = vec![0i16; max_samples];
    let m = decoder
        .decode_into(enc_buf, &mut dec_buf)
        .expect("decode_into");
    assert_eq!(m, pcm.len(), "decoded length mismatch");
    for (a, b) in dec_buf[..m].iter().zip(pcm.iter()) {
        let diff = (*a as i32 - *b as i32).abs();
        assert!(diff <= tol, "round-trip diff {} exceeds tol {}", diff, tol);
    }
}

#[test]
fn test_buffer_too_small_is_reported() {
    let mut enc = pcmu::PcmuEncoder::new();
    let pcm = [0i16; 16];
    let mut tiny = [0u8; 4];
    let err = enc.encode_into(&pcm, &mut tiny).unwrap_err();
    assert_eq!(err, CodecError::BufferTooSmall);
}

#[test]
fn test_pcmu_into_roundtrip() {
    let pcm: Vec<i16> = (0..128).map(|i| (i * 100) as i16).collect();
    // μ-law quantization step is up to ~256 in the worst segment.
    roundtrip_approx(
        &mut pcmu::PcmuEncoder::new(),
        &mut pcmu::PcmuDecoder::new(),
        &pcm,
        256,
    );
}

#[test]
fn test_pcma_into_roundtrip() {
    let pcm: Vec<i16> = (0..128).map(|i| ((i - 64) * 100) as i16).collect();
    // A-law quantization step is up to ~256 in the worst segment.
    roundtrip_approx(
        &mut pcma::PcmaEncoder::new(),
        &mut pcma::PcmaDecoder::new(),
        &pcm,
        256,
    );
}

#[test]
fn test_g722_into_roundtrip() {
    // G.722 16kHz mode: 320 samples = 20ms. Encoder emits 160 bytes.
    let pcm: Vec<i16> = (0..320)
        .map(|i| (((i as f64) * 0.05).sin() * 10000.0) as i16)
        .collect();
    let mut enc = g722::G722Encoder::new();
    let mut dec = g722::G722Decoder::new();

    let max_bytes = enc.max_encode_bytes(pcm.len());
    let mut enc_buf = vec![0u8; max_bytes];
    let n = enc.encode_into(&pcm, &mut enc_buf).expect("encode_into");
    assert_eq!(n, 160, "G.722 16kHz 20ms should produce 160 bytes");

    let max_samples = dec.max_decode_samples(n);
    let mut dec_buf = vec![0i16; max_samples];
    let m = dec
        .decode_into(&enc_buf[..n], &mut dec_buf)
        .expect("decode_into");
    assert_eq!(m, 320, "G.722 16kHz 160 bytes should decode to 320 samples");

    // Bit-exact round-trip is not expected from a lossy codec, but the decoded
    // signal should have reasonable energy.
    let energy: f64 = dec_buf[..m]
        .iter()
        .map(|&s| (s as f64).powi(2))
        .sum::<f64>()
        / m as f64;
    assert!(
        energy.sqrt() > 100.0,
        "decoded G.722 has no energy: {}",
        energy
    );
}

#[test]
fn test_g722_max_sizes_are_correct() {
    let enc = g722::G722Encoder::new();
    assert_eq!(enc.max_encode_bytes(320), 161); // n/2 + 1
    assert_eq!(enc.max_encode_bytes(0), 1);

    let dec = g722::G722Decoder::new();
    assert_eq!(dec.max_decode_samples(160), 320); // n*2
}

#[test]
fn test_g729_into_roundtrip() {
    // G.729 8kHz, 80 samples = 10ms frame, encoded to 10 bytes.
    let pcm: Vec<i16> = (0..80)
        .map(|i| (((i as f64) * 0.1).sin() * 8000.0) as i16)
        .collect();
    let mut enc = g729::G729Encoder::new();
    let mut dec = g729::G729Decoder::new();

    let max_bytes = enc.max_encode_bytes(pcm.len());
    assert_eq!(
        max_bytes, 10,
        "G.729 80 samples should produce 10 bytes max"
    );
    let mut enc_buf = vec![0u8; max_bytes];
    let n = enc.encode_into(&pcm, &mut enc_buf).expect("encode_into");
    assert_eq!(n, 10);

    let max_samples = dec.max_decode_samples(n);
    assert_eq!(max_samples, 80);
    let mut dec_buf = vec![0i16; max_samples];
    let m = dec
        .decode_into(&enc_buf[..n], &mut dec_buf)
        .expect("decode_into");
    assert_eq!(m, 80);
}

#[test]
fn test_byte_sample_helpers_into() {
    let samples: Vec<i16> = vec![0x0102, -0x0304, 0x7fff, -0x8000];
    let mut bytes = vec![0u8; samples.len() * 2];
    let n = samples_to_bytes_into(&samples, &mut bytes).expect("s2b");
    assert_eq!(n, samples.len() * 2);

    let mut back = vec![0i16; samples.len()];
    let m = bytes_to_samples_into(&bytes, &mut back).expect("b2s");
    assert_eq!(m, samples.len());
    assert_eq!(&back[..m], &samples[..]);
}

#[test]
fn test_byte_helpers_buffer_too_small() {
    let samples = vec![0i16; 4];
    let mut bytes = vec![0u8; 4]; // too small (need 8)
    let err = samples_to_bytes_into(&samples, &mut bytes).unwrap_err();
    assert_eq!(err, CodecError::BufferTooSmall);
}

#[test]
fn test_byte_helpers_empty_inputs() {
    // Regression: an empty `out` slice has a dangling 1-aligned pointer, which
    // used to trip the `from_raw_parts_mut` alignment precondition.
    let n = samples_to_bytes_into(&[], &mut []).expect("s2b empty");
    assert_eq!(n, 0);
    let m = bytes_to_samples_into(&[], &mut []).expect("b2s empty");
    assert_eq!(m, 0);
}

#[test]
fn test_byte_helpers_unaligned_buffer() {
    // Regression: a `&mut [u8]` at an odd address is not `i16`-aligned.
    let samples: Vec<i16> = vec![0x0102, -0x0304, 0x7fff, -0x8000];

    let mut buf = vec![0u8; samples.len() * 2 + 1];
    let odd = &mut buf[1..];
    assert!(
        odd.as_ptr().addr() % 2 == 1,
        "test requires an odd-addressed slice"
    );
    let n = samples_to_bytes_into(&samples, odd).expect("s2b unaligned");
    assert_eq!(n, samples.len() * 2);

    let mut back = vec![0i16; samples.len()];
    let m = bytes_to_samples_into(&odd[..n], &mut back).expect("b2s unaligned");
    assert_eq!(m, samples.len());
    assert_eq!(&back[..m], &samples[..]);
}

#[test]
fn test_resampler_into_roundtrip_via_borrowed_coeffs() {
    use audio_codec::resampler::{COEFFS_LEN, Resampler};

    let mut coeffs = vec![0.0f32; COEFFS_LEN];
    let mut r = Resampler::new(8000, 16000, &mut coeffs).expect("resampler init");

    let input: Vec<i16> = (0..80).map(|i| (i * 100) as i16).collect();
    let max = r.max_output_samples(input.len());
    let mut out = vec![0i16; max];
    let n = r.resample_into(&input, &mut out).expect("resample_into");
    // 8k -> 16k should roughly double the sample count.
    assert!(
        n >= 150 && n <= 170,
        "expected ~160 output samples, got {}",
        n
    );

    // Calling resample_into with an undersized buffer must return BufferTooSmall.
    let mut tiny = [0i16; 4];
    let err = r.resample_into(&input, &mut tiny).unwrap_err();
    assert_eq!(err, CodecError::BufferTooSmall);
}

#[test]
fn test_resampler_rejects_undersized_coeffs() {
    use audio_codec::resampler::Resampler;

    let mut too_small = vec![0.0f32; 100];
    let res = Resampler::new(8000, 16000, &mut too_small);
    assert!(matches!(res, Err(CodecError::BufferTooSmall)));
}

#[test]
fn test_resampler_rejects_zero_rate() {
    use audio_codec::resampler::Resampler;

    let mut coeffs = vec![0.0f32; audio_codec::resampler::COEFFS_LEN];
    let res = Resampler::new(0, 8000, &mut coeffs);
    assert!(matches!(res, Err(CodecError::InvalidInput)));
}

#[test]
fn test_try_from_returns_codec_error() {
    use audio_codec::CodecType;

    let err = CodecType::try_from(99u8).unwrap_err();
    assert_eq!(err, CodecError::InvalidCodecType);

    let err = CodecType::try_from("nonsense").unwrap_err();
    assert_eq!(err, CodecError::InvalidCodecName);
}
