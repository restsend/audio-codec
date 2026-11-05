#[cfg(feature = "std")]
use super::PcmBuf;
use super::{CodecError, Decoder, Encoder, Sample};
use crate::fade::{self, RECOVER_MS};
// `Box` lives in the std prelude; under `no_std` it comes from `alloc`.
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
pub use opus_rs::Application as OpusApplication;
use opus_rs::{Application, OpusDecoder as OpusDecoderRaw, OpusEncoder as OpusEncoderRaw};

// Heap-free scratch-buffer caps (worst case: 48 kHz, stereo, 20 ms frame).
const OPUS_MAX_FRAME: usize = 960; // 20 ms @ 48 kHz per channel
const OPUS_MAX_CHANNELS: usize = 2;
const OPUS_MAX_SAMPLES: usize = OPUS_MAX_FRAME * OPUS_MAX_CHANNELS; // 1920
// RFC 6716: a single Opus packet carries at most 1276 bytes. Used by the
// std-only `encode` helper's scratch packet buffer.
#[cfg(feature = "std")]
const OPUS_MAX_PACKET: usize = 1276;

pub struct OpusDecoder {
    decoder: Box<OpusDecoderRaw>,
    sample_rate: u32,
    channels: u16,
    w_output_f32: [f32; OPUS_MAX_SAMPLES],
    w_pcm_i16: [i16; OPUS_MAX_SAMPLES],
    /// Samples per channel concealed in the current loss; 0 when not
    /// concealing. Opus's concealment carries on near the level of the
    /// audio before the loss, so it's faded here (see [`crate::fade`]).
    concealed: usize,
}

impl OpusDecoder {
    /// Create a new Opus decoder instance
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        let decoder = Box::new(
            OpusDecoderRaw::new(sample_rate as i32, channels as usize)
                .expect("Failed to create Opus decoder"),
        );

        Self {
            decoder,
            sample_rate,
            channels,
            w_output_f32: [0.0; OPUS_MAX_SAMPLES],
            w_pcm_i16: [0; OPUS_MAX_SAMPLES],
            concealed: 0,
        }
    }

    /// Create a default Opus decoder (48kHz, stereo)
    pub fn new_default() -> Self {
        Self::new(48000, 2)
    }

    /// Lower-level decode that does not perform the stereo→mono downmix.
    /// Used internally by both the trait impl and the back-comat `decode`.
    pub fn decode_into_raw(&mut self, data: &[u8], output: &mut [i16]) -> usize {
        if data.is_empty() {
            return 0;
        }

        // Detect the actual channel count from the Opus packet's TOC byte
        // (bit 2 is the stereo flag per RFC 6716). opus-rs 0.1.19+ rejects
        // decoding when the packet channel count doesn't match the decoder's,
        // so we adapt the decoder here to avoid returning empty PCM.
        let packet_channels = if data[0] & 0x04 != 0 { 2usize } else { 1 };
        if self.channels as usize != packet_channels {
            self.channels = packet_channels as u16;
            // Reuse the existing heap allocation (see `clippy::replace_box`).
            *self.decoder = OpusDecoderRaw::new(self.sample_rate as i32, packet_channels)
                .expect("Failed to create Opus decoder");
            self.concealed = 0;
        }

        let channels = usize::from(self.channels);
        let frame_size = (self.sample_rate as usize * 20) / 1000;
        let max_samples = frame_size * channels;

        match self
            .decoder
            .decode(data, frame_size, &mut self.w_output_f32[..max_samples])
        {
            Ok(len) => {
                let total_samples = len * channels;
                if total_samples == 0 {
                    return 0;
                }

                for i in 0..total_samples {
                    self.w_pcm_i16[i] =
                        (self.w_output_f32[i] * 32768.0).clamp(-32768.0, 32767.0) as i16;
                }

                let n = total_samples.min(output.len());
                output[..n].copy_from_slice(&self.w_pcm_i16[..n]);
                // a DTX frame (the TOC byte alone) is the decoder's comfort
                // noise, not the end of a loss
                if data.len() > 1 {
                    self.end_loss(&mut output[..n], channels);
                }
                n
            }
            Err(_) => 0,
        }
    }
}

impl OpusDecoder {
    /// Conceal a lost frame of `samples` samples per channel with Opus's own
    /// packet loss concealment or, given `next`, rebuild it from the in-band
    /// FEC of the packet after it; without the stereo→mono downmix.
    fn conceal_into_raw(
        &mut self,
        next: Option<&[u8]>,
        samples: usize,
        out: &mut [Sample],
    ) -> Result<usize, CodecError> {
        let channels = usize::from(self.channels);
        let total = samples * channels;
        if total > OPUS_MAX_SAMPLES {
            return Err(CodecError::InvalidInput);
        }
        if out.len() < total {
            return Err(CodecError::BufferTooSmall);
        }
        let pcm = &mut self.w_output_f32[..total];
        let len = match next {
            Some(next) => self.decoder.decode_fec(next, samples, pcm),
            // opus-rs conceals a packet of 0 or 1 bytes, as libopus does
            // with `data == NULL`
            None => self.decoder.decode(&[], samples, pcm),
        }
        .map_err(|_| CodecError::DecodeFailed)?;
        let n = len * channels;
        for (o, f) in out[..n].iter_mut().zip(&self.w_output_f32[..n]) {
            *o = (f * 32768.0).clamp(-32768.0, 32767.0) as i16;
        }
        match next {
            // rebuilt from the sender's own data: audio like a decoded
            // packet's (where `next` carried no FEC, the frame is concealed
            // at full level, as the first of a loss would be)
            Some(_) => self.end_loss(&mut out[..n], channels),
            None => self.fade(&mut out[..n], channels),
        }
        Ok(n)
    }

    /// Fade a concealed frame by how far into the loss it is.
    fn fade(&mut self, out: &mut [Sample], channels: usize) {
        let channels = channels.max(1);
        for (i, frame) in out.chunks_mut(channels).enumerate() {
            let gain = fade::gain(self.concealed + i, self.sample_rate);
            for s in frame {
                *s = (*s as i32 * gain / 100) as Sample;
            }
        }
        self.concealed += out.len() / channels;
    }

    /// Bring decoded audio ending a loss back in from where the fade had
    /// got to, over `RECOVER_MS`.
    fn end_loss(&mut self, out: &mut [Sample], channels: usize) {
        if self.concealed == 0 {
            return;
        }
        let from = fade::gain(self.concealed, self.sample_rate);
        self.concealed = 0;
        if from >= 100 {
            return;
        }
        let channels = channels.max(1);
        let len = (self.sample_rate * RECOVER_MS / 1000) as usize;
        let total = (len + 1) as i32;
        for (i, frame) in out.chunks_mut(channels).take(len).enumerate() {
            let w = (i + 1) as i32;
            let gain = from + (100 - from) * w / total;
            for s in frame {
                *s = (*s as i32 * gain / 100) as Sample;
            }
        }
    }

    /// `conceal_into_raw` with the stereo→mono downmix `decode` does.
    #[cfg(feature = "std")]
    fn conceal_downmixed(&mut self, next: Option<&[u8]>, samples: usize) -> PcmBuf {
        let mut pcm = vec![0i16; samples * usize::from(self.channels)];
        match self.conceal_into_raw(next, samples, &mut pcm) {
            Ok(n) => pcm.truncate(n),
            Err(_) => return Vec::new(),
        }
        if usize::from(self.channels) == 2 {
            pcm = pcm
                .chunks_exact(2)
                .map(|chunk| ((chunk[0] as i32 + chunk[1] as i32) / 2) as i16)
                .collect();
        }
        pcm
    }
}

impl Decoder for OpusDecoder {
    /// Opus's concealment is part of the decoder: always on.
    fn enable_concealment(&mut self) -> bool {
        true
    }

    /// Conceals with Opus's own packet loss concealment, faded through a
    /// long loss as G.711 and G.722's is: full level for 10 ms, silence at
    /// 60 ms. `samples` must be an Opus frame size (a multiple of 2.5 ms, at
    /// most 20 ms); others fail with `DecodeFailed`.
    fn conceal_into(&mut self, samples: usize, out: &mut [Sample]) -> Result<usize, CodecError> {
        self.conceal_into_raw(None, samples, out)
    }

    /// Override the default to downmix stereo to mono, as `decode` does.
    #[cfg(feature = "std")]
    fn conceal(&mut self, samples: usize) -> PcmBuf {
        self.conceal_downmixed(None, samples)
    }

    /// Rebuilds the lost frame from `next`'s in-band FEC (libopus's
    /// `decode_fec`), concealing what it doesn't cover. `samples` as for
    /// `conceal_into`.
    fn recover_into(
        &mut self,
        next: &[u8],
        samples: usize,
        out: &mut [Sample],
    ) -> Result<usize, CodecError> {
        self.conceal_into_raw(Some(next), samples, out)
    }

    /// Override the default to downmix stereo to mono, as `decode` does.
    #[cfg(feature = "std")]
    fn recover(&mut self, next: &[u8], samples: usize) -> PcmBuf {
        self.conceal_downmixed(Some(next), samples)
    }

    fn decode_into(&mut self, data: &[u8], out: &mut [Sample]) -> Result<usize, CodecError> {
        if data.is_empty() {
            return Ok(0);
        }
        let n = self.decode_into_raw(data, out);
        if n == 0 {
            Err(CodecError::DecodeFailed)
        } else {
            Ok(n)
        }
    }

    fn max_decode_samples(&self, n_bytes: usize) -> usize {
        // One 20ms frame per packet is the typical case; size for stereo
        // to cover the worst case (caller may downmix afterwards).
        let frame_size = (self.sample_rate as usize * 20) / 1000;
        let _ = n_bytes;
        frame_size * 2
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    /// Override the default to preserve back-comat stereo→mono downmix
    /// when the decoder was configured with `channels == 2`.
    #[cfg(feature = "std")]
    fn decode(&mut self, data: &[u8]) -> PcmBuf {
        if data.is_empty() {
            return Vec::new();
        }
        let packet_channels = if data[0] & 0x04 != 0 { 2usize } else { 1 };

        let frame_size = (self.sample_rate as usize * 20) / 1000;
        let max_samples = frame_size * packet_channels;
        let mut pcm = vec![0i16; max_samples];
        let n = self.decode_into_raw(data, &mut pcm);
        pcm.truncate(n);
        if usize::from(self.channels) == 2 {
            pcm = pcm
                .chunks_exact(2)
                .map(|chunk| ((chunk[0] as i32 + chunk[1] as i32) / 2) as i16)
                .collect();
        }
        pcm
    }
}

pub struct OpusEncoder {
    encoder: Box<OpusEncoderRaw>,
    sample_rate: u32,
    channels: u16,
    w_input_f32: [f32; OPUS_MAX_SAMPLES],
}

impl OpusEncoder {
    pub fn new_with_application(sample_rate: u32, channels: u16, application: Application) -> Self {
        let encoder = Box::new(
            OpusEncoderRaw::new(sample_rate as i32, channels as usize, application)
                .expect("Failed to create Opus encoder"),
        );

        Self {
            encoder,
            sample_rate,
            channels,
            w_input_f32: [0.0; OPUS_MAX_SAMPLES],
        }
    }

    /// Create a new Opus encoder instance.
    ///
    /// Keep backward-compatible defaults with pre-0.3.31 behavior:
    /// - VoIP application
    /// - caller can provide mono PCM even when encoder is configured as stereo;
    ///   `encode()` duplicates mono samples to stereo.
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        let mut enc = Self::new_with_application(sample_rate, channels, Application::Voip);
        enc.encoder.bitrate_bps = if channels == 2 { 64000 } else { 48000 };
        enc.encoder.complexity = 5;
        enc.encoder.use_cbr = true;
        enc
    }

    /// Create a default Opus encoder (48kHz, stereo)
    pub fn new_default() -> Self {
        Self::new(48000, 2)
    }

    /// Set the encoder bitrate in bits per second.
    pub fn set_bitrate(&mut self, bitrate_bps: i32) {
        self.encoder.bitrate_bps = bitrate_bps;
    }

    /// Set the encoder complexity (0-10).
    pub fn set_complexity(&mut self, complexity: i32) {
        self.encoder.complexity = complexity;
    }

    /// Enable or disable constant bitrate (CBR) mode.
    pub fn set_cbr(&mut self, cbr: bool) {
        self.encoder.use_cbr = cbr;
    }

    /// Enable or disable in-band FEC: SILK frames also carry a low-bitrate
    /// copy of the previous frame, which a decoder can rebuild a lost packet
    /// from (`Decoder::recover_into`). Only coded when the expected packet
    /// loss (`set_packet_loss_perc`) is above zero and the bitrate leaves
    /// room for it (e.g. 32 kb/s for wideband speech).
    pub fn set_inband_fec(&mut self, enabled: bool) {
        self.encoder.use_inband_fec = enabled;
    }

    /// Set the packet loss the encoder expects, in percent (0-100). It
    /// trades bitrate for robustness, and gates in-band FEC.
    pub fn set_packet_loss_perc(&mut self, percent: i32) {
        self.encoder.packet_loss_perc = percent.clamp(0, 100);
    }

    /// Encode into a caller-provided packet buffer.
    ///
    /// Returns `Some(bytes_written)` on success. Expects `samples` to match
    /// the encoder's channel count (interleaved stereo when `channels == 2`).
    pub fn encode_into_raw(&mut self, samples: &[Sample], output: &mut [u8]) -> Option<usize> {
        let channels = usize::from(self.channels);
        if samples.is_empty() || channels == 0 || !samples.len().is_multiple_of(channels) {
            return None;
        }
        // Fixed scratch cap (heap-free build).
        if samples.len() > OPUS_MAX_SAMPLES {
            return None;
        }

        let frame_size = samples.len() / channels;

        for (dst, &s) in self.w_input_f32[..samples.len()]
            .iter_mut()
            .zip(samples.iter())
        {
            *dst = s as f32 / 32768.0;
        }

        self.encoder
            .encode(&self.w_input_f32[..samples.len()], frame_size, output)
            .ok()
    }

    #[cfg(feature = "std")]
    fn encode_raw(&mut self, samples: &[Sample]) -> Vec<u8> {
        let channels = usize::from(self.channels);
        if samples.is_empty()
            || channels == 0
            || !samples.len().is_multiple_of(channels)
            || samples.len() > OPUS_MAX_SAMPLES
        {
            return Vec::new();
        }

        let frame_size = samples.len() / channels;

        for (dst, &s) in self.w_input_f32[..samples.len()]
            .iter_mut()
            .zip(samples.iter())
        {
            *dst = s as f32 / 32768.0;
        }

        let mut packet = [0u8; OPUS_MAX_PACKET];
        match self
            .encoder
            .encode(&self.w_input_f32[..samples.len()], frame_size, &mut packet)
        {
            Ok(len) => packet[..len].to_vec(),
            Err(_) => Vec::new(),
        }
    }
}

impl Encoder for OpusEncoder {
    fn encode_into(&mut self, samples: &[Sample], out: &mut [u8]) -> Result<usize, CodecError> {
        if samples.is_empty() {
            return Ok(0);
        }
        // Note: this expects `samples` to already match the encoder's channel
        // count (interleaved stereo if `channels == 2`). For the legacy
        // mono→stereo upmix behavior, use the std-only `encode` method.
        match self.encode_into_raw(samples, out) {
            Some(n) => Ok(n),
            None => Err(CodecError::EncodeFailed),
        }
    }

    fn max_encode_bytes(&self, _n_samples: usize) -> usize {
        // Per RFC 6716 the maximum Opus packet size is 1275 bytes.
        1275
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    /// Override the default to preserve back-comat mono→stereo upmix
    /// when the encoder was configured with `channels == 2`.
    #[cfg(feature = "std")]
    fn encode(&mut self, samples: &[Sample]) -> Vec<u8> {
        if self.channels == 2 {
            // mono→stereo upmix into a local scratch buffer (no allocation).
            if samples.len() > OPUS_MAX_FRAME {
                return Vec::new();
            }
            let mut stereo = [0i16; OPUS_MAX_SAMPLES];
            for (i, &sample) in samples.iter().enumerate() {
                stereo[2 * i] = sample;
                stereo[2 * i + 1] = sample;
            }
            return self.encode_raw(&stereo[..samples.len() * 2]);
        }
        self.encode_raw(samples)
    }
}

// The tests below exercise the std-only allocating `encode`/`decode`
// conveniences (including the stereo-downmix handling), so they only compile
// with `std`. The `*_into` trait API is covered portably by
// `test_opus_into_roundtrip` at the bottom, which also runs in `no_std`.
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn make_mono_pcm_20ms() -> Vec<i16> {
        (0..960).map(|i| ((i * 100) % 32767) as i16).collect()
    }

    /// 20 ms frames of a continuous 220 Hz tone at 48 kHz.
    fn tone_frame(frame: usize) -> Vec<i16> {
        (0..960)
            .map(|i| {
                let t = (frame * 960 + i) as f32 / 48000.0;
                (8000.0 * (2.0 * core::f32::consts::PI * 220.0 * t).sin()) as i16
            })
            .collect()
    }

    fn rms(s: &[i16]) -> f64 {
        (s.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
    }

    #[test]
    fn test_opus_conceals_a_lost_frame_with_its_own_plc() {
        let mut enc = OpusEncoder::new_default();
        let mut dec = OpusDecoder::new_default();
        for f in 0..10 {
            dec.decode(&enc.encode(&tone_frame(f)));
        }
        // frame 10 is lost
        let concealed = dec.conceal(960);
        assert_eq!(concealed.len(), 960, "downmixed like decode");
        assert!(
            rms(&concealed) > 1000.0,
            "concealment is silent: rms {}",
            rms(&concealed)
        );
        // the stream carries on after the loss
        enc.encode(&tone_frame(10));
        let next = dec.decode(&enc.encode(&tone_frame(11)));
        assert_eq!(next.len(), 960);
        assert!(rms(&next) > 1000.0);
    }

    /// Voiced, speech-like 48 kHz mono: 8 harmonics of a pitch gliding
    /// 120 → 180 Hz under a slow envelope.
    fn voice(frames: usize) -> Vec<Vec<i16>> {
        let mut phase = 0f32;
        let pcm: Vec<i16> = (0..frames * 960)
            .map(|i| {
                let t = i as f32 / 48000.0;
                let f0 = 120.0 + 60.0 * (t / 0.6).min(1.0);
                phase += 2.0 * core::f32::consts::PI * f0 / 48000.0;
                let env = 0.6 + 0.4 * (2.0 * core::f32::consts::PI * 2.0 * t).sin();
                let v: f32 = (1..=8).map(|k| (k as f32 * phase).sin() / k as f32).sum();
                (6000.0 * env * v) as i16
            })
            .collect();
        pcm.chunks(960).map(<[i16]>::to_vec).collect()
    }

    #[test]
    fn test_opus_recovers_a_lost_frame_from_inband_fec() {
        // speech at 32 kb/s (SILK or Hybrid), told to expect loss so it codes
        // FEC; at 16 kb/s fullband Hybrid it codes none
        let mut enc = OpusEncoder::new(48000, 1);
        enc.set_bitrate(32000);
        enc.set_inband_fec(true);
        enc.set_packet_loss_perc(20);
        let packets: Vec<Vec<u8>> = voice(40).iter().map(|f| enc.encode(f)).collect();
        // SILK or Hybrid: FEC lives in the SILK layer (TOC configs 0-15)
        let has_silk = |p: &[u8]| p[0] >> 3 < 16;
        assert!(packets.iter().all(|p| has_silk(p)), "coded without SILK");

        let mut reference = OpusDecoder::new(48000, 1);
        let clean: Vec<Vec<i16>> = packets.iter().map(|p| reference.decode(p)).collect();
        let error_db = |out: &[i16], clean: &[i16]| {
            let err: f64 = out
                .iter()
                .zip(clean)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            let sig: f64 = clean.iter().map(|b| (*b as f64).powi(2)).sum();
            10.0 * (err / sig).log10()
        };
        let (mut recovered_db, mut concealed_db) = (0f64, 0f64);
        let losses = [10, 17, 23, 31];
        for &lost in &losses {
            let mut recovering = OpusDecoder::new(48000, 1);
            let mut concealing = OpusDecoder::new(48000, 1);
            for p in &packets[..lost] {
                recovering.decode(p);
                concealing.decode(p);
            }
            let recovered = recovering.recover(&packets[lost + 1], 960);
            let concealed = concealing.conceal(960);
            assert_eq!(recovered.len(), 960);
            recovered_db += error_db(&recovered, &clean[lost]);
            concealed_db += error_db(&concealed, &clean[lost]);
            // `next` is not consumed: it decodes as usual afterwards
            assert_eq!(recovering.decode(&packets[lost + 1]).len(), 960);
        }
        let n = losses.len() as f64;
        let (recovered_db, concealed_db) = (recovered_db / n, concealed_db / n);
        println!(
            "lost frame error: recovered {recovered_db:.1} dB, concealed {concealed_db:.1} dB"
        );
        // measured: recovered -11.7 dB, concealed -5.0 dB
        assert!(
            recovered_db < concealed_db - 4.0,
            "FEC no better than concealment: {recovered_db:.1} vs {concealed_db:.1} dB"
        );
    }

    #[test]
    fn test_opus_concealment_fades_through_a_long_loss() {
        let mut enc = OpusEncoder::new(48000, 1);
        let packets: Vec<Vec<u8>> = voice(12).iter().map(|f| enc.encode(f)).collect();
        let mut dec = OpusDecoder::new(48000, 1);
        for p in &packets[..10] {
            dec.decode(p);
        }
        let levels: Vec<f64> = (0..5).map(|_| rms(&dec.conceal(960))).collect();
        // full for 10 ms, then 20% per 10 ms: silent from 60 ms
        assert!(
            levels[0] > 500.0,
            "first frame of a loss is audible: {levels:?}"
        );
        assert!(
            levels[1] < levels[0] && levels[2] < levels[1],
            "not fading: {levels:?}"
        );
        assert_eq!(levels[3..], [0.0, 0.0], "not silent from 60 ms: {levels:?}");
    }

    #[test]
    fn test_opus_decoding_after_a_faded_loss_ramps_in() {
        let mut enc = OpusEncoder::new(48000, 1);
        let packets: Vec<Vec<u8>> = voice(12).iter().map(|f| enc.encode(f)).collect();
        let mut dec = OpusDecoder::new(48000, 1);
        // opus-rs on its own, given the same calls
        let mut raw = OpusDecoderRaw::new(48000, 1).unwrap();
        let mut raw_out = [0f32; 960];
        for p in &packets[..10] {
            dec.decode(p);
            raw.decode(p, 960, &mut raw_out).unwrap();
        }
        for _ in 0..5 {
            dec.conceal(960);
            raw.decode(&[], 960, &mut raw_out).unwrap();
        }
        let resumed = dec.decode(&packets[10]);
        raw.decode(&packets[10], 960, &mut raw_out).unwrap();
        let expected: Vec<i16> = raw_out
            .iter()
            .map(|f| (f * 32768.0).clamp(-32768.0, 32767.0) as i16)
            .collect();
        // from silence over 4 ms, then as decoded
        let ramp = 48000 * RECOVER_MS as usize / 1000;
        assert!(resumed[0].unsigned_abs() <= expected[0].unsigned_abs() / 100 + 1);
        assert_eq!(resumed[ramp..], expected[ramp..]);
        // the loss is over: the next one starts at full level
        dec.conceal(960);
        assert_eq!(dec.concealed, 960);
    }

    #[test]
    fn test_opus_recover_without_fec_conceals() {
        let mut enc = OpusEncoder::new(48000, 1);
        let packets: Vec<Vec<u8>> = voice(12).iter().map(|f| enc.encode(f)).collect();
        let mut dec = OpusDecoder::new(48000, 1);
        for p in &packets[..10] {
            dec.decode(p);
        }
        // no FEC coded: the lost frame is concealed, still audible
        let recovered = dec.recover(&packets[11], 960);
        assert_eq!(recovered.len(), 960);
        assert!(rms(&recovered) > 500.0, "rms {}", rms(&recovered));
    }

    #[test]
    fn test_opus_conceal_into_is_interleaved_and_checks_the_buffer() {
        let mut enc = OpusEncoder::new_default();
        let mut dec = OpusDecoder::new_default();
        dec.decode(&enc.encode(&tone_frame(0)));
        let mut out = [0i16; 1920];
        // stereo decoder: 960 samples per channel, interleaved
        assert_eq!(dec.conceal_into(960, &mut out), Ok(1920));
        let mut small = [0i16; 100];
        assert_eq!(
            dec.conceal_into(960, &mut small),
            Err(CodecError::BufferTooSmall)
        );
    }

    #[test]
    fn test_opus_encode_decode_20ms_produces_960_mono_samples() {
        let mut enc = OpusEncoder::new_default();
        let pcm = make_mono_pcm_20ms();
        let opus_pkt = enc.encode(&pcm);
        assert!(!opus_pkt.is_empty(), "encoder should produce output");

        // TOC byte bit 2 should be set (stereo flag from mono→stereo upmix)
        assert!(opus_pkt[0] & 0x04 != 0, "packet should be stereo");

        let mut dec = OpusDecoder::new_default();
        let decoded = dec.decode(&opus_pkt);

        // 20ms at 48kHz mono = 960 samples
        assert_eq!(
            decoded.len(),
            960,
            "decoder should downmix stereo→mono and output 960 samples, got {}",
            decoded.len()
        );
    }

    #[test]
    fn test_opus_consecutive_frames_all_produce_960() {
        let mut enc = OpusEncoder::new_default();
        let pcm = make_mono_pcm_20ms();
        let opus_pkt = enc.encode(&pcm);

        let mut dec = OpusDecoder::new_default();
        for i in 0..5 {
            let decoded = dec.decode(&opus_pkt);
            assert_eq!(
                decoded.len(),
                960,
                "frame {} should produce 960 mono samples, got {}",
                i,
                decoded.len()
            );
        }
    }

    #[test]
    fn test_opus_decoder_output_has_reasonable_energy() {
        let mut enc = OpusEncoder::new_default();
        // 440Hz sine at 48kHz, 20ms
        let pcm: Vec<i16> = (0..960)
            .map(|i| {
                let t = i as f64 / 48000.0;
                (16384.0 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as i16
            })
            .collect();
        let opus_pkt = enc.encode(&pcm);

        let mut dec = OpusDecoder::new_default();
        let decoded = dec.decode(&opus_pkt);
        assert_eq!(decoded.len(), 960);

        let energy: f64 =
            decoded.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / decoded.len() as f64;
        let rms = energy.sqrt();
        assert!(
            rms > 100.0 && rms < 20000.0,
            "decoded audio RMS {} should be in reasonable range",
            rms
        );
    }

    #[test]
    fn test_opus_decoder_handles_mono_packet_gracefully() {
        // Create a mono Opus encoder, encode PCM → mono Opus packet
        let mut mono_enc = OpusEncoder::new(48000, 1);
        let pcm = make_mono_pcm_20ms();
        let mono_pkt = mono_enc.encode(&pcm);
        assert!(!mono_pkt.is_empty());
        // Mono packet TOC bit 2 should be 0
        assert!(
            mono_pkt[0] & 0x04 == 0,
            "mono encoder should produce mono packet"
        );

        // Decode with default stereo decoder — should handle mono→stereo transition
        let mut dec = OpusDecoder::new_default();
        let decoded = dec.decode(&mono_pkt);
        assert_eq!(
            decoded.len(),
            960,
            "first mono packet should produce exactly 960 mono samples, got {}",
            decoded.len()
        );

        // Second mono packet should also be 960
        let decoded2 = dec.decode(&mono_pkt);
        assert_eq!(
            decoded2.len(),
            960,
            "second mono packet should also produce 960 mono samples, got {}",
            decoded2.len()
        );
    }

    #[test]
    fn test_opus_stereo_packet_downmix_produces_960() {
        let mut enc = OpusEncoder::new_default();
        let pcm = make_mono_pcm_20ms();
        let stereo_pkt = enc.encode(&pcm);
        assert!(stereo_pkt[0] & 0x04 != 0);

        let mut dec = OpusDecoder::new_default();
        let decoded = dec.decode(&stereo_pkt);
        assert_eq!(
            decoded.len(),
            960,
            "stereo packet should downmix to 960 mono samples"
        );
    }
}

/// Portable roundtrip through the trait `encode_into`/`decode_into` API
/// (compiles with or without `std`; the no_std opus path).
#[cfg(all(test, feature = "opus"))]
mod no_std_tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_opus_into_roundtrip() {
        let mut enc = OpusEncoder::new_default();
        let mut dec = OpusDecoder::new(48000, 1);
        let pcm: Vec<i16> = (0..960).map(|i| ((i * 100) % 32767) as i16).collect();

        let mut packet = alloc::vec![0u8; enc.max_encode_bytes(pcm.len())];
        let n = enc
            .encode_into(&pcm, &mut packet)
            .expect("packet sized by max_encode_bytes");
        packet.truncate(n);
        assert!(!packet.is_empty(), "encoder should produce output");

        let mut out = alloc::vec![0i16; dec.max_decode_samples(packet.len())];
        let samples = dec
            .decode_into(&packet, &mut out)
            .expect("out sized by max_decode_samples");
        assert_eq!(samples, 960, "20 ms at 48 kHz must decode to 960 samples");
        let rms: f64 = out[..samples]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / samples as f64;
        assert!(rms.sqrt() > 1000.0, "decoded signal should keep energy");
    }
}
