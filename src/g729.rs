use super::{CodecError, Decoder, Encoder, Sample};
use crate::fade;

const L_FRAME: usize = 80; // 10ms frame at 8kHz
const L_FRAME_COMPRESSED: usize = 10; // G.729 frame size in bytes
/// An Annex B SID frame, ending a payload after any speech frames
/// (RFC 3551 §4.5.6).
const L_SID: usize = 2;

/// G.729 audio decoder using g729-sys
pub struct G729Decoder {
    decoder: g729_sys::Decoder,
    /// G.729's frame erasure concealment (§4.4) decays on its own but
    /// settles some 30 dB down rather than at silence; it's faded on top
    /// like the other codecs' (see [`crate::fade`]).
    loss: fade::LossFade,
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
            loss: fade::LossFade::default(),
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

        // 10-byte speech frames of 80 samples each, then possibly a SID
        // frame, which decodes to 80 samples of comfort noise (Annex B)
        if out.len() < self.max_decode_samples(data.len()) {
            return Err(CodecError::BufferTooSmall);
        }
        let mut written = 0usize;
        let (frames, rest) = data.as_chunks::<L_FRAME_COMPRESSED>();
        for frame in frames {
            // g729-sys: decode(payload, frame_erased, is_sid, rfc3389)
            let decoded = self.decoder.decode(frame, false, false, false);
            out[written..written + L_FRAME].copy_from_slice(&decoded);
            written += L_FRAME;
        }
        if rest.len() == L_SID {
            let decoded = self.decoder.decode(rest, false, true, false);
            out[written..written + L_FRAME].copy_from_slice(&decoded);
            written += L_FRAME;
        }
        self.loss.end_loss(&mut out[..written], 1, 8000);

        Ok(written)
    }

    fn max_decode_samples(&self, n_bytes: usize) -> usize {
        let sid = usize::from(n_bytes % L_FRAME_COMPRESSED == L_SID);
        (n_bytes / L_FRAME_COMPRESSED + sid) * L_FRAME
    }

    fn sample_rate(&self) -> u32 {
        8000 // G.729 operates at 8kHz
    }

    fn channels(&self) -> u16 {
        1 // G.729 is always mono
    }

    /// G.729's concealment is part of the decoder: always on.
    fn enable_concealment(&mut self) -> bool {
        true
    }

    /// Conceals with G.729's frame erasure concealment (§4.4): the last
    /// spectral envelope and pitch carry on with their gains attenuated,
    /// the decoder's predictors kept in step, so the packets after the loss
    /// decode close to what the sender encoded. In DTX, after a SID frame,
    /// it carries on the comfort noise instead (Annex B). Faded through a
    /// long loss like the other codecs: full level for 10 ms, silence at
    /// 60 ms. `samples` must be a multiple of 80 (10 ms); others fail with
    /// `InvalidInput`.
    fn conceal_into(&mut self, samples: usize, out: &mut [Sample]) -> Result<usize, CodecError> {
        if !samples.is_multiple_of(L_FRAME) {
            return Err(CodecError::InvalidInput);
        }
        if out.len() < samples {
            return Err(CodecError::BufferTooSmall);
        }
        for frame in out[..samples].as_chunks_mut::<L_FRAME>().0 {
            *frame = self.decoder.decode(&[], true, false, false);
        }
        self.loss.fade(&mut out[..samples], 1, 8000);
        Ok(samples)
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
