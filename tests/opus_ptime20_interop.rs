use std::f64::consts::PI;

use audio_codec::opus::{OpusDecoder, OpusEncoder};
use audio_codec::{Decoder, Encoder};
use opusic_sys::{
    OPUS_APPLICATION_AUDIO, OPUS_OK, OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST,
    OPUS_SET_VBR_REQUEST, OpusDecoder as CDecoder, OpusEncoder as CEncoder, opus_decode,
    opus_decoder_create, opus_decoder_destroy, opus_encode, opus_encoder_create,
    opus_encoder_ctl, opus_encoder_destroy,
};

const OUTPUT_RATE: usize = 48_000;
const CHANNELS: usize = 2;
const FRAME_MS: usize = 20;
const FRAME_SIZE: usize = OUTPUT_RATE * FRAME_MS / 1000;
const FRAME_SAMPLES: usize = FRAME_SIZE * CHANNELS;
const TEST_FRAMES: usize = 120;
const WARMUP_FRAMES: usize = 8;
const MIN_SNR_DB: f64 = 8.0;
const MIN_CORR: f64 = 0.70;

struct CEncoderWrap {
    ptr: *mut CEncoder,
    channels: usize,
}

impl CEncoderWrap {
    fn new(sample_rate: i32, channels: usize) -> Self {
        let mut err = 0;
        let ptr = unsafe {
            opus_encoder_create(
                sample_rate,
                channels as i32,
                OPUS_APPLICATION_AUDIO,
                &mut err,
            )
        };
        assert!(err == OPUS_OK && !ptr.is_null(), "create opusic encoder failed");
        unsafe {
            let _ = opus_encoder_ctl(ptr, OPUS_SET_BITRATE_REQUEST, 64_000);
            let _ = opus_encoder_ctl(ptr, OPUS_SET_VBR_REQUEST, 0);
            let _ = opus_encoder_ctl(ptr, OPUS_SET_COMPLEXITY_REQUEST, 9);
        }
        Self { ptr, channels }
    }

    fn encode(&mut self, pcm: &[i16]) -> Vec<u8> {
        let mut packet = vec![0u8; 1500];
        let frame_size = (pcm.len() / self.channels) as i32;
        let n = unsafe {
            opus_encode(
                self.ptr,
                pcm.as_ptr(),
                frame_size,
                packet.as_mut_ptr(),
                packet.len() as i32,
            )
        };
        assert!(n > 0, "opusic encode failed: {n}");
        packet.truncate(n as usize);
        packet
    }
}

impl Drop for CEncoderWrap {
    fn drop(&mut self) {
        unsafe { opus_encoder_destroy(self.ptr) }
    }
}

struct CDecoderWrap {
    ptr: *mut CDecoder,
    channels: usize,
}

impl CDecoderWrap {
    fn new(sample_rate: i32, channels: usize) -> Self {
        let mut err = 0;
        let ptr = unsafe { opus_decoder_create(sample_rate, channels as i32, &mut err) };
        assert!(err == OPUS_OK && !ptr.is_null(), "create opusic decoder failed");
        Self { ptr, channels }
    }

    fn decode(&mut self, packet: &[u8], frame_size: usize) -> Vec<i16> {
        let mut out = vec![0i16; frame_size * self.channels];
        let n = unsafe {
            opus_decode(
                self.ptr,
                packet.as_ptr(),
                packet.len() as i32,
                out.as_mut_ptr(),
                frame_size as i32,
                0,
            )
        };
        assert!(n > 0, "opusic decode failed: {n}");
        out.truncate(n as usize * self.channels);
        out
    }
}

impl Drop for CDecoderWrap {
    fn drop(&mut self) {
        unsafe { opus_decoder_destroy(self.ptr) }
    }
}

#[test]
fn ptime20_interop_audio_codec_and_opusic() {
    let input = gen_test_stereo_48k();
    let ac_packets = encode_audio_codec_packets(&input);
    let c_packets = encode_opusic_packets(&input);

    assert!(!ac_packets.is_empty());
    assert_eq!(ac_packets.len(), c_packets.len());

    let chains = [
        ("audio-codec -> audio-codec", decode_audio_codec_packets(&ac_packets)),
        ("audio-codec -> opusic", decode_opusic_packets(&ac_packets)),
        ("opusic -> audio-codec", decode_audio_codec_packets(&c_packets)),
        ("opusic -> opusic", decode_opusic_packets(&c_packets)),
    ];

    for (label, output) in chains {
        assert!(!output.is_empty(), "{label} output is empty");
        assert!(output.len() >= input.len() / 2, "{label} output too short");
        let (snr_l, corr_l, snr_r, corr_r) = measure_stereo_quality(&input, &output);
        assert!(snr_l >= MIN_SNR_DB, "{label} left SNR too low: {snr_l:.2}");
        assert!(snr_r >= MIN_SNR_DB, "{label} right SNR too low: {snr_r:.2}");
        assert!(corr_l >= MIN_CORR, "{label} left corr too low: {corr_l:.4}");
        assert!(corr_r >= MIN_CORR, "{label} right corr too low: {corr_r:.4}");
    }
}

fn gen_test_stereo_48k() -> Vec<i16> {
    let wanted = TEST_FRAMES * FRAME_SAMPLES;
    let total_samples = wanted + WARMUP_FRAMES * FRAME_SAMPLES;
    let mut pcm = Vec::with_capacity(total_samples);
    let mut phase_l = 0.0f64;
    let mut phase_r = 0.0f64;
    let mut prng_l = 0x1234_5678u32;
    let mut prng_r = 0x9abc_def0u32;

    for n in 0..(total_samples / CHANNELS) {
        let t = n as f64 / OUTPUT_RATE as f64;
        let speech_env = if t < 0.35 { t / 0.35 } else { 1.0 };

        let f_l = 250.0 + 900.0 * (0.13 * t).sin().abs();
        let f_r = 300.0 + 1200.0 * (0.11 * t + 0.7).sin().abs();
        phase_l += 2.0 * PI * f_l / OUTPUT_RATE as f64;
        phase_r += 2.0 * PI * f_r / OUTPUT_RATE as f64;

        prng_l = prng_l.wrapping_mul(1664525).wrapping_add(1013904223);
        prng_r = prng_r.wrapping_mul(22695477).wrapping_add(1);
        let noise_l = ((prng_l >> 8) as f64 / ((1u64 << 24) as f64) - 0.5) * 0.08;
        let noise_r = ((prng_r >> 8) as f64 / ((1u64 << 24) as f64) - 0.5) * 0.08;

        let l = (phase_l.sin() * 0.42 + (phase_l * 0.37).sin() * 0.18 + noise_l) * speech_env;
        let r = (phase_r.sin() * 0.40 + (phase_r * 0.29).sin() * 0.16 + noise_r) * speech_env;
        pcm.push((l * 32767.0) as i16);
        pcm.push((r * 32767.0) as i16);
    }
    pcm.truncate(total_samples);
    pcm
}

fn encode_audio_codec_packets(input: &[i16]) -> Vec<Vec<u8>> {
    let mut encoder = OpusEncoder::new(OUTPUT_RATE as u32, CHANNELS as u16);
    input
        .chunks_exact(FRAME_SAMPLES)
        .map(|frame| {
            let packet = encoder.encode(frame);
            assert!(!packet.is_empty(), "audio-codec encode returned empty packet");
            packet
        })
        .collect()
}

fn decode_audio_codec_packets(packets: &[Vec<u8>]) -> Vec<i16> {
    let mut decoder = OpusDecoder::new(OUTPUT_RATE as u32, CHANNELS as u16);
    let mut out = Vec::new();
    for packet in packets {
        let frame = decoder.decode(packet);
        if !frame.is_empty() {
            out.extend_from_slice(&frame);
        }
    }
    out
}

fn encode_opusic_packets(input: &[i16]) -> Vec<Vec<u8>> {
    let mut encoder = CEncoderWrap::new(OUTPUT_RATE as i32, CHANNELS);
    input
        .chunks_exact(FRAME_SAMPLES)
        .map(|frame| encoder.encode(frame))
        .collect()
}

fn decode_opusic_packets(packets: &[Vec<u8>]) -> Vec<i16> {
    let mut decoder = CDecoderWrap::new(OUTPUT_RATE as i32, CHANNELS);
    let mut out = Vec::new();
    for packet in packets {
        out.extend_from_slice(&decoder.decode(packet, FRAME_SIZE));
    }
    out
}

fn measure_stereo_quality(reference: &[i16], test: &[i16]) -> (f64, f64, f64, f64) {
    let warmup = WARMUP_FRAMES * FRAME_SIZE;
    let max_lag = FRAME_SIZE;
    let ref_l = channel_slice(reference, 0);
    let ref_r = channel_slice(reference, 1);
    let test_l = channel_slice(test, 0);
    let test_r = channel_slice(test, 1);
    let lag = find_best_lag(&ref_l, &test_l, max_lag, warmup);
    (
        snr_db(&ref_l, &test_l, lag, warmup),
        corr(&ref_l, &test_l, lag, warmup),
        snr_db(&ref_r, &test_r, lag, warmup),
        corr(&ref_r, &test_r, lag, warmup),
    )
}

fn channel_slice(stereo: &[i16], channel: usize) -> Vec<i16> {
    stereo.chunks_exact(2).map(|x| x[channel]).collect()
}

fn find_best_lag(a: &[i16], b: &[i16], max_lag: usize, start: usize) -> usize {
    let mut best_lag = 0;
    let mut best_score = f64::MIN;
    for lag in 0..=max_lag.min(b.len().saturating_sub(1)) {
        let end = a.len().min(b.len().saturating_sub(lag));
        let mut score = 0.0;
        let mut count = 0usize;
        for i in start..end {
            score += a[i] as f64 * b[i + lag] as f64;
            count += 1;
        }
        if count > 0 {
            score /= count as f64;
        }
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }
    best_lag
}

fn snr_db(a: &[i16], b: &[i16], lag: usize, start: usize) -> f64 {
    let end = a.len().min(b.len().saturating_sub(lag));
    if start >= end {
        return -999.0;
    }
    let mut sig = 0.0;
    let mut err = 0.0;
    for i in start..end {
        let x = a[i] as f64;
        let y = b[i + lag] as f64;
        sig += x * x;
        let d = x - y;
        err += d * d;
    }
    if err < 1e-9 {
        120.0
    } else {
        10.0 * (sig / err).log10()
    }
}

fn corr(a: &[i16], b: &[i16], lag: usize, start: usize) -> f64 {
    let end = a.len().min(b.len().saturating_sub(lag));
    if start >= end {
        return 0.0;
    }
    let n = (end - start) as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in start..end {
        let x = a[i] as f64;
        let y = b[i + lag] as f64;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    let num = sxy - sx * sy / n;
    let den_x = sxx - sx * sx / n;
    let den_y = syy - sy * sy / n;
    if den_x <= 1e-9 || den_y <= 1e-9 {
        0.0
    } else {
        num / (den_x.sqrt() * den_y.sqrt())
    }
}
