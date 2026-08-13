use super::{CodecError, Decoder, Encoder, Sample};

const L_FRAME: usize = 80; // 10ms frame at 8kHz
const L_FRAME_COMPRESSED: usize = 10; // G.729 frame size in bytes

/// G.729 audio decoder using g729-sys
pub struct G729Decoder {
    decoder: g729_sys::Decoder,
}

impl Default for G729Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl G729Decoder {
    /// Create a new G.729 decoder instance.
    pub fn new() -> Self {
        Self {
            decoder: g729_sys::Decoder::new(),
        }
    }
}

unsafe impl Send for G729Decoder {}
unsafe impl Sync for G729Decoder {}

impl Decoder for G729Decoder {
    fn decode_into(&mut self, data: &[u8], out: &mut [Sample]) -> Result<usize, CodecError> {
        if data.is_empty() {
            return Ok(0);
        }

        // G.729 processes 10-byte frames, each producing 80 samples
        let mut written = 0usize;
        let mut pos = 0usize;

        while pos + L_FRAME_COMPRESSED <= data.len() {
            if out.len() < written + L_FRAME {
                return Err(CodecError::BufferTooSmall);
            }
            let frame_data = &data[pos..pos + L_FRAME_COMPRESSED];
            // g729-sys: decode(frame, bfi, vad, dtx) -> [i16;80]
            let decoded_frame = self.decoder.decode(frame_data, false, false, false);
            out[written..written + L_FRAME].copy_from_slice(&decoded_frame);
            written += L_FRAME;
            pos += L_FRAME_COMPRESSED;
        }

        Ok(written)
    }

    fn max_decode_samples(&self, n_bytes: usize) -> usize {
        (n_bytes / L_FRAME_COMPRESSED) * L_FRAME
    }

    fn sample_rate(&self) -> u32 {
        8000 // G.729 operates at 8kHz
    }

    fn channels(&self) -> u16 {
        1 // G.729 is always mono
    }
}

/// G.729 audio encoder using g729-sys
pub struct G729Encoder {
    encoder: g729_sys::Encoder,
}

impl Default for G729Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl G729Encoder {
    /// Create a new G.729 encoder instance.
    ///
    /// Annex B (VAD/DTX) is disabled by default.
    pub fn new() -> Self {
        Self {
            encoder: g729_sys::Encoder::new(false),
        }
    }

    /// Create a new G.729 encoder with explicit VAD/DTX control.
    pub fn with_vad(enable_vad: bool) -> Self {
        Self {
            encoder: g729_sys::Encoder::new(enable_vad),
        }
    }
}

unsafe impl Send for G729Encoder {}
unsafe impl Sync for G729Encoder {}

impl Encoder for G729Encoder {
    fn encode_into(&mut self, samples: &[Sample], out: &mut [u8]) -> Result<usize, CodecError> {
        if samples.is_empty() {
            return Ok(0);
        }

        let mut written = 0usize;
        let mut pos = 0usize;
        let mut frame_arr = [0i16; L_FRAME];
        let mut packet = [0u8; L_FRAME_COMPRESSED];

        while pos + L_FRAME <= samples.len() {
            frame_arr.copy_from_slice(&samples[pos..pos + L_FRAME]);
            let n = self.encoder.encode_into(&frame_arr, &mut packet);
            let n = n as usize;
            if out.len() < written + n {
                return Err(CodecError::BufferTooSmall);
            }
            out[written..written + n].copy_from_slice(&packet[..n]);
            written += n;
            pos += L_FRAME;
        }

        Ok(written)
    }

    fn max_encode_bytes(&self, n_samples: usize) -> usize {
        (n_samples / L_FRAME) * L_FRAME_COMPRESSED
    }

    fn sample_rate(&self) -> u32 {
        8000 // G.729 operates at 8kHz
    }

    fn channels(&self) -> u16 {
        1 // G.729 is always mono
    }
}
