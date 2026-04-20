use super::{Decoder, Encoder, PcmBuf, Sample};
pub use opus_rs::Application as OpusApplication;
use opus_rs::{Application, OpusDecoder as OpusDecoderRaw, OpusEncoder as OpusEncoderRaw};

pub struct OpusDecoder {
    decoder: OpusDecoderRaw,
    sample_rate: u32,
    channels: u16,
    w_output_f32: Vec<f32>,
    w_pcm_i16: Vec<i16>,
}

impl OpusDecoder {
    /// Create a new Opus decoder instance
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        let decoder = OpusDecoderRaw::new(sample_rate as i32, channels as usize)
            .expect("Failed to create Opus decoder");

        Self {
            decoder,
            sample_rate,
            channels,
            w_output_f32: Vec::new(),
            w_pcm_i16: Vec::new(),
        }
    }

    /// Create a default Opus decoder (48kHz, stereo)
    pub fn new_default() -> Self {
        Self::new(48000, 2)
    }

    pub fn decode_into(&mut self, data: &[u8], output: &mut [i16]) -> usize {
        let channels = usize::from(self.channels);
        if channels == 0 || data.is_empty() {
            return 0;
        }

        // opus-rs currently handles up to 20ms frames reliably; use that as the decode frame size
        let frame_size = (self.sample_rate as usize * 20) / 1000;
        let max_samples = frame_size * channels;
        if self.w_output_f32.len() < max_samples {
            self.w_output_f32.resize(max_samples, 0.0);
        }

        match self
            .decoder
            .decode(data, frame_size, &mut self.w_output_f32[..max_samples])
        {
            Ok(len) => {
                let total_samples = len * channels;
                if total_samples == 0 {
                    return 0;
                }

                if self.w_pcm_i16.len() < total_samples {
                    self.w_pcm_i16.resize(total_samples, 0);
                }

                for i in 0..total_samples {
                    self.w_pcm_i16[i] =
                        (self.w_output_f32[i] * 32768.0).clamp(-32768.0, 32767.0) as i16;
                }

                let n = total_samples.min(output.len());
                output[..n].copy_from_slice(&self.w_pcm_i16[..n]);
                n
            }
            Err(_) => 0,
        }
    }
}

impl Decoder for OpusDecoder {
    fn decode(&mut self, data: &[u8]) -> PcmBuf {
        let channels = usize::from(self.channels);
        if channels == 0 || data.is_empty() {
            return Vec::new();
        }

        let frame_size = (self.sample_rate as usize * 20) / 1000;
        let max_samples = frame_size * channels;
        let mut pcm = vec![0i16; max_samples];
        let n = self.decode_into(data, &mut pcm);
        pcm.truncate(n);
        pcm
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }
}

pub struct OpusEncoder {
    encoder: OpusEncoderRaw,
    sample_rate: u32,
    channels: u16,
    w_input_f32: Vec<f32>,
    w_packet: Vec<u8>,
}

impl OpusEncoder {
    pub fn new_with_application(sample_rate: u32, channels: u16, application: Application) -> Self {
        let encoder = OpusEncoderRaw::new(sample_rate as i32, channels as usize, application)
            .expect("Failed to create Opus encoder");

        Self {
            encoder,
            sample_rate,
            channels,
            w_input_f32: Vec::new(),
            w_packet: vec![0u8; 1275],
        }
    }

    /// Create a new Opus encoder instance.
    ///
    /// For stereo input, prefer `Application::Audio` by default.
    /// `opus-rs` currently has stability issues on some 48k stereo VoIP paths.
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        let app = if channels == 2 {
            Application::Audio
        } else {
            Application::Voip
        };
        let mut enc = Self::new_with_application(sample_rate, channels, app);
        enc.encoder.bitrate_bps = if channels == 2 { 64000 } else { 48000 };
        enc.encoder.complexity = if channels == 2 { 5 } else { 0 };
        enc
    }

    /// Create a default Opus encoder (48kHz, stereo, Audio mode, 64kbps)
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

    /// Encode into a caller-provided packet buffer.
    ///
    /// Returns `Some(bytes_written)` on success.
    pub fn encode_into(&mut self, samples: &[Sample], output: &mut [u8]) -> Option<usize> {
        let channels = usize::from(self.channels);
        if samples.is_empty() || channels == 0 || samples.len() % channels != 0 {
            return None;
        }

        let frame_size = samples.len() / channels;

        if self.w_input_f32.len() < samples.len() {
            self.w_input_f32.resize(samples.len(), 0.0);
        }
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

    fn encode_raw(&mut self, samples: &[Sample]) -> Vec<u8> {
        let channels = usize::from(self.channels);
        if samples.is_empty() || channels == 0 || samples.len() % channels != 0 {
            return Vec::new();
        }

        let frame_size = samples.len() / channels;

        if self.w_input_f32.len() < samples.len() {
            self.w_input_f32.resize(samples.len(), 0.0);
        }
        for (dst, &s) in self.w_input_f32[..samples.len()]
            .iter_mut()
            .zip(samples.iter())
        {
            *dst = s as f32 / 32768.0;
        }

        match self.encoder.encode(
            &self.w_input_f32[..samples.len()],
            frame_size,
            &mut self.w_packet,
        ) {
            Ok(len) => {
                let mut out = Vec::with_capacity(len);
                out.extend_from_slice(&self.w_packet[..len]);
                out
            }
            Err(_) => Vec::new(),
        }
    }
}

impl Encoder for OpusEncoder {
    fn encode(&mut self, samples: &[Sample]) -> Vec<u8> {
        self.encode_raw(samples)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }
}
