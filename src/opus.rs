use super::{Decoder, Encoder, PcmBuf, Sample};

use opus_rs::{OpusDecoder as OpusDecoderRaw, OpusEncoder as OpusEncoderRaw, Application};

/// Opus audio decoder backed by opus-rs
pub struct OpusDecoder {
    decoder: OpusDecoderRaw,
    sample_rate: u32,
    channels: u16,
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
        }
    }

    /// Create a default Opus decoder (48kHz, stereo)
    pub fn new_default() -> Self {
        #[cfg(feature = "opus_mono")]
        return Self::new(48000, 1);

        #[cfg(not(feature = "opus_mono"))]
        Self::new(48000, 2)
    }
}

impl Decoder for OpusDecoder {
    fn decode(&mut self, data: &[u8]) -> PcmBuf {
        let channels = usize::from(self.channels);
        if channels == 0 {
            return Vec::new();
        }

        // Allow up to 120ms of audio: 48kHz * 0.12s * 2 channels = 11520 samples
        let max_samples = 11520;
        let mut output = vec![0f32; max_samples];

        let result = self.decoder.decode(data, max_samples / channels, &mut output);

        match result {
            Ok(len) => {
                let total_samples = len * channels;
                output.truncate(total_samples);

                // Convert f32 to i16
                let pcm: Vec<i16> = output
                    .iter()
                    .map(|&s| (s * 32768.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();

                // If stereo, convert to mono
                if channels == 2 {
                    pcm.chunks_exact(2)
                        .map(|chunk| ((chunk[0] as i32 + chunk[1] as i32) / 2) as i16)
                        .collect()
                } else {
                    pcm
                }
            }
            Err(_) => Vec::new(),
        }
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }
}

/// Opus audio encoder backed by opus-rs
pub struct OpusEncoder {
    encoder: OpusEncoderRaw,
    sample_rate: u32,
    channels: u16,
}

impl OpusEncoder {
    /// Create a new Opus encoder instance
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        let encoder = OpusEncoderRaw::new(
            sample_rate as i32,
            channels as usize,
            Application::Voip,
        )
        .expect("Failed to create Opus encoder");

        Self {
            encoder,
            sample_rate,
            channels,
        }
    }

    /// Create a default Opus encoder (48kHz, stereo)
    pub fn new_default() -> Self {
        #[cfg(feature = "opus_mono")]
        return Self::new(48000, 1);

        #[cfg(not(feature = "opus_mono"))]
        Self::new(48000, 2)
    }

    fn encode_raw(&mut self, samples: &[Sample]) -> Vec<u8> {
        let channels = usize::from(self.channels);
        if samples.is_empty() || channels == 0 || samples.len() % channels != 0 {
            return Vec::new();
        }

        let frame_size = samples.len() / channels;

        // Convert i16 samples to f32
        let input: Vec<f32> = samples
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();

        // Estimate max output size: 1 byte per 80 samples (worst case)
        let max_output_size = (frame_size / 20).max(1) + 2;
        let mut output = vec![0u8; max_output_size];

        match self.encoder.encode(&input, frame_size, &mut output) {
            Ok(len) => {
                output.truncate(len);
                output
            }
            Err(_) => Vec::new(),
        }
    }
}

impl Encoder for OpusEncoder {
    fn encode(&mut self, samples: &[Sample]) -> Vec<u8> {
        if self.channels == 2 {
            let mut stereo_samples = Vec::with_capacity(samples.len() * 2);
            for &sample in samples {
                stereo_samples.push(sample);
                stereo_samples.push(sample);
            }
            return self.encode_raw(&stereo_samples);
        }

        self.encode_raw(samples)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }
}
