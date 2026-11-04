#![cfg_attr(not(feature = "std"), no_std)]

// Opus wraps the (large, ~260 KB) opus-rs encoder/decoder on the heap, so it
// needs `alloc` even in `no_std`. The telephone-event payload layer (and any
// other Vec-returning helper) uses it in BOTH std and no_std builds.
extern crate alloc;

pub use error::CodecError;

pub mod error;
#[cfg(any(feature = "plc", feature = "opus"))]
mod fade;
pub mod g722;
#[cfg(feature = "g729")]
pub mod g729;
#[cfg(feature = "opus")]
pub mod opus;
pub mod pcma;
pub mod pcmu;
#[cfg(feature = "plc")]
mod plc;
/// Default polyphase resampler (f32 core, exact integer timing).
pub mod resampler;
/// Alternative polyphase resampler with a Q15 fixed-point core, for targets
/// without an FPU. Not the default; use [`resampler`] unless you need it.
pub mod resampler_fixed;
pub mod telephone_event;

#[cfg(feature = "std")]
pub use resampler::Resampler;
#[cfg(feature = "std")]
pub use resampler::{BoxedResampler, resample};

pub type Sample = i16;

#[cfg(feature = "std")]
pub type PcmBuf = Vec<Sample>;

#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub enum CodecType {
    PCMU,
    PCMA,
    G722,
    #[cfg(feature = "g729")]
    G729,
    #[cfg(feature = "opus")]
    Opus,
    TelephoneEvent,
}

/// Decoder trait: converts codec-specific bytes into PCM samples.
///
/// The slice-based `decode_into` is the primary, always-available method
/// (works in `no_std`). The convenience `decode` returning a `Vec` is only
/// available with the `std` feature and has a default implementation that
/// delegates to `decode_into`.
pub trait Decoder: Send + Sync {
    /// Decode `data` into `out`, returning the number of samples written.
    ///
    /// Returns [`CodecError::BufferTooSmall`] if `out` cannot hold the
    /// decoded samples (use `max_decode_samples` to size it).
    fn decode_into(&mut self, data: &[u8], out: &mut [Sample]) -> Result<usize, CodecError>;

    /// Upper bound on the number of samples that `decode_into` will write
    /// for an input of `n_bytes` bytes.
    fn max_decode_samples(&self, n_bytes: usize) -> usize;

    /// Get the sample rate of the decoded audio.
    fn sample_rate(&self) -> u32;

    /// Get the number of channels.
    fn channels(&self) -> u16;

    /// Turn on the decoder's own packet loss concealment, returning whether
    /// [`Decoder::conceal_into`] will conceal. Off by default where keeping
    /// it ready costs something on every decoded packet: G.711 and G.722
    /// keep a history of the decoded audio, and have concealment only with
    /// the `plc` feature. Opus's is part of the decoder and always on. The
    /// default returns `false`: no concealment.
    fn enable_concealment(&mut self) -> bool {
        false
    }

    /// Conceal a lost packet: write `samples` samples per channel of audio
    /// to play in its place into `out`, returning the number written
    /// (`samples * channels`). The decoder's state carries on through the
    /// loss as the codec allows, and the next packet decoded after it is
    /// blended in. A long loss fades out: full level for 10 ms, silence at
    /// 60 ms (after G.711 Appendix I).
    ///
    /// Call it once per lost packet, in order, between the `decode_into`
    /// calls of the packets around the loss. Returns `Ok(0)`, writing
    /// nothing, when the decoder has no concealment or it isn't enabled
    /// (see [`Decoder::enable_concealment`]): the caller has to conceal the
    /// gap itself.
    fn conceal_into(&mut self, samples: usize, out: &mut [Sample]) -> Result<usize, CodecError> {
        let _ = (samples, out);
        Ok(0)
    }

    /// Rebuild a lost packet from `next`, the packet that followed it, when
    /// the codec carries redundancy for it (Opus in-band FEC), into `out`
    /// like [`Decoder::conceal_into`]. `next` is not consumed: decode it
    /// with `decode_into` afterwards.
    ///
    /// Use it in place of `conceal_into` for the last packet of a loss when
    /// the packet after it has already arrived (a jitter buffer usually has
    /// it). Where the codec has no redundancy, or `next` carries none, this
    /// conceals; the default is just `conceal_into`.
    fn recover_into(
        &mut self,
        next: &[u8],
        samples: usize,
        out: &mut [Sample],
    ) -> Result<usize, CodecError> {
        let _ = next;
        self.conceal_into(samples, out)
    }

    /// Convenience wrapper that allocates a `Vec` and calls `recover_into`.
    /// Empty when the decoder can neither recover nor conceal.
    #[cfg(feature = "std")]
    fn recover(&mut self, next: &[u8], samples: usize) -> PcmBuf {
        let mut buf = vec![0i16; samples * self.channels() as usize];
        match self.recover_into(next, samples, &mut buf) {
            Ok(n) => {
                buf.truncate(n);
                buf
            }
            Err(_) => Vec::new(),
        }
    }

    /// Convenience wrapper that allocates a `Vec` and calls `conceal_into`.
    /// Empty when the decoder has no concealment of its own.
    #[cfg(feature = "std")]
    fn conceal(&mut self, samples: usize) -> PcmBuf {
        let mut buf = vec![0i16; samples * self.channels() as usize];
        match self.conceal_into(samples, &mut buf) {
            Ok(n) => {
                buf.truncate(n);
                buf
            }
            Err(_) => Vec::new(),
        }
    }

    /// Convenience wrapper that allocates a `Vec` and calls `decode_into`.
    #[cfg(feature = "std")]
    fn decode(&mut self, data: &[u8]) -> PcmBuf {
        let max = self.max_decode_samples(data.len());
        let mut buf = vec![0i16; max];
        match self.decode_into(data, &mut buf) {
            Ok(n) => {
                buf.truncate(n);
                buf
            }
            Err(_) => Vec::new(),
        }
    }
}

/// Encoder trait: converts PCM samples into codec-specific bytes.
///
/// The slice-based `encode_into` is the primary, always-available method
/// (works in `no_std`). The convenience `encode` returning a `Vec` is only
/// available with the `std` feature and has a default implementation that
/// delegates to `encode_into`.
pub trait Encoder: Send + Sync {
    /// Encode `samples` into `out`, returning the number of bytes written.
    ///
    /// Returns [`CodecError::BufferTooSmall`] if `out` cannot hold the
    /// encoded bytes (use `max_encode_bytes` to size it).
    fn encode_into(&mut self, samples: &[Sample], out: &mut [u8]) -> Result<usize, CodecError>;

    /// Upper bound on the number of bytes that `encode_into` will write
    /// for an input of `n_samples` samples.
    fn max_encode_bytes(&self, n_samples: usize) -> usize;

    /// Get the sample rate expected for input samples.
    fn sample_rate(&self) -> u32;

    /// Get the number of channels expected for input.
    fn channels(&self) -> u16;

    /// Convenience wrapper that allocates a `Vec` and calls `encode_into`.
    #[cfg(feature = "std")]
    fn encode(&mut self, samples: &[Sample]) -> Vec<u8> {
        let max = self.max_encode_bytes(samples.len());
        let mut buf = vec![0u8; max];
        match self.encode_into(samples, &mut buf) {
            Ok(n) => {
                buf.truncate(n);
                buf
            }
            Err(_) => Vec::new(),
        }
    }
}

#[cfg(feature = "std")]
pub fn create_decoder(codec: CodecType) -> Box<dyn Decoder> {
    match codec {
        CodecType::PCMU => Box::new(pcmu::PcmuDecoder::new()),
        CodecType::PCMA => Box::new(pcma::PcmaDecoder::new()),
        CodecType::G722 => Box::new(g722::G722Decoder::new()),
        #[cfg(feature = "g729")]
        CodecType::G729 => Box::new(g729::G729Decoder::new()),
        #[cfg(feature = "opus")]
        CodecType::Opus => Box::new(opus::OpusDecoder::new_default()),
        CodecType::TelephoneEvent => Box::new(telephone_event::TelephoneEventDecoder::new()),
    }
}

#[cfg(feature = "std")]
pub fn create_encoder(codec: CodecType) -> Box<dyn Encoder> {
    match codec {
        CodecType::PCMU => Box::new(pcmu::PcmuEncoder::new()),
        CodecType::PCMA => Box::new(pcma::PcmaEncoder::new()),
        CodecType::G722 => Box::new(g722::G722Encoder::new()),
        #[cfg(feature = "g729")]
        CodecType::G729 => Box::new(g729::G729Encoder::new()),
        #[cfg(feature = "opus")]
        CodecType::Opus => Box::new(opus::OpusEncoder::new_default()),
        CodecType::TelephoneEvent => Box::new(telephone_event::TelephoneEventEncoder::new()),
    }
}

#[cfg(all(feature = "std", feature = "opus"))]
pub fn create_opus_encoder(
    sample_rate: u32,
    channels: u16,
    application: opus::OpusApplication,
) -> Box<dyn Encoder> {
    Box::new(opus::OpusEncoder::new_with_application(
        sample_rate,
        channels,
        application,
    ))
}

#[cfg(all(feature = "std", feature = "opus"))]
pub fn create_opus_decoder(sample_rate: u32, channels: u16) -> Box<dyn Decoder> {
    Box::new(opus::OpusDecoder::new(sample_rate, channels))
}

impl CodecType {
    pub fn mime_type(&self) -> &str {
        match self {
            CodecType::PCMU => "audio/PCMU",
            CodecType::PCMA => "audio/PCMA",
            CodecType::G722 => "audio/G722",
            #[cfg(feature = "g729")]
            CodecType::G729 => "audio/G729",
            #[cfg(feature = "opus")]
            CodecType::Opus => "audio/opus",
            CodecType::TelephoneEvent => "audio/telephone-event",
        }
    }
    pub fn rtpmap(&self) -> &str {
        match self {
            CodecType::PCMU => "PCMU/8000",
            CodecType::PCMA => "PCMA/8000",
            CodecType::G722 => "G722/8000",
            #[cfg(feature = "g729")]
            CodecType::G729 => "G729/8000",
            #[cfg(feature = "opus")]
            CodecType::Opus => "opus/48000/2",
            CodecType::TelephoneEvent => "telephone-event/8000",
        }
    }
    pub fn fmtp(&self) -> Option<&str> {
        match self {
            CodecType::PCMU => None,
            CodecType::PCMA => None,
            CodecType::G722 => None,
            #[cfg(feature = "g729")]
            CodecType::G729 => None,
            #[cfg(feature = "opus")]
            CodecType::Opus => Some("minptime=10;useinbandfec=1;stereo=1;sprop-stereo=1"),
            CodecType::TelephoneEvent => Some("0-16"),
        }
    }

    pub fn clock_rate(&self) -> u32 {
        match self {
            CodecType::PCMU => 8000,
            CodecType::PCMA => 8000,
            CodecType::G722 => 8000,
            #[cfg(feature = "g729")]
            CodecType::G729 => 8000,
            #[cfg(feature = "opus")]
            CodecType::Opus => 48000,
            CodecType::TelephoneEvent => 8000,
        }
    }

    pub fn channels(&self) -> u16 {
        match self {
            #[cfg(feature = "opus")]
            CodecType::Opus => 2,
            _ => 1,
        }
    }

    pub fn payload_type(&self) -> u8 {
        match self {
            CodecType::PCMU => 0,
            CodecType::PCMA => 8,
            CodecType::G722 => 9,
            #[cfg(feature = "g729")]
            CodecType::G729 => 18,
            #[cfg(feature = "opus")]
            CodecType::Opus => 111,
            CodecType::TelephoneEvent => 101,
        }
    }
    pub fn samplerate(&self) -> u32 {
        match self {
            CodecType::PCMU => 8000,
            CodecType::PCMA => 8000,
            CodecType::G722 => 16000,
            #[cfg(feature = "g729")]
            CodecType::G729 => 8000,
            #[cfg(feature = "opus")]
            CodecType::Opus => 48000,
            CodecType::TelephoneEvent => 8000,
        }
    }
    pub fn is_audio(&self) -> bool {
        match self {
            CodecType::PCMU | CodecType::PCMA | CodecType::G722 => true,
            #[cfg(feature = "g729")]
            CodecType::G729 => true,
            #[cfg(feature = "opus")]
            CodecType::Opus => true,
            _ => false,
        }
    }

    /// Whether `payload` says the sender went quiet: it is, or ends with,
    /// the codec's silence frame under discontinuous transmission (DTX).
    /// A sender in DTX then sends nothing, or only more of these, until
    /// speech resumes, so a jitter buffer should take the frames missing
    /// after one as silence rather than loss, and not conceal them.
    ///
    /// - Opus: a DTX packet, at most 2 bytes (as WebRTC tells them apart).
    /// - G.729: a payload ending with an Annex B SID frame, 2 bytes after
    ///   any 10-byte speech frames (RFC 3551 §4.5.6).
    /// - Others have no DTX of their own: `false`. Comfort noise sent in its
    ///   own payload type (RFC 3389 `CN`, static payload type 13) means the
    ///   same, but isn't a codec here: check for it by payload type.
    pub fn is_silence_frame(&self, payload: &[u8]) -> bool {
        match self {
            #[cfg(feature = "opus")]
            CodecType::Opus => !payload.is_empty() && payload.len() <= 2,
            #[cfg(feature = "g729")]
            CodecType::G729 => payload.len() % 10 == 2,
            _ => {
                let _ = payload;
                false
            }
        }
    }

    pub fn is_dynamic(&self) -> bool {
        match self {
            #[cfg(feature = "opus")]
            CodecType::Opus => true,
            CodecType::TelephoneEvent => true,
            _ => false,
        }
    }
}

impl TryFrom<u8> for CodecType {
    type Error = CodecError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(CodecType::PCMU),
            8 => Ok(CodecType::PCMA),
            9 => Ok(CodecType::G722),
            #[cfg(feature = "g729")]
            18 => Ok(CodecType::G729), // Static payload type
            // Dynamic payload type should get from the rtpmap in sdp offer, leave this for backward compatibility
            101 => Ok(CodecType::TelephoneEvent),
            #[cfg(feature = "opus")]
            111 => Ok(CodecType::Opus), // Dynamic payload type
            _ => Err(CodecError::InvalidCodecType),
        }
    }
}

impl TryFrom<&str> for CodecType {
    type Error = CodecError;

    fn try_from(name: &str) -> Result<Self, Self::Error> {
        let b = name.as_bytes();
        if b.eq_ignore_ascii_case(b"pcmu") || b.eq_ignore_ascii_case(b"ulaw") {
            Ok(CodecType::PCMU)
        } else if b.eq_ignore_ascii_case(b"pcma") || b.eq_ignore_ascii_case(b"alaw") {
            Ok(CodecType::PCMA)
        } else if b.eq_ignore_ascii_case(b"g722") {
            Ok(CodecType::G722)
        } else if cfg!(feature = "g729") && b.eq_ignore_ascii_case(b"g729") {
            #[cfg(feature = "g729")]
            {
                Ok(CodecType::G729)
            }
            #[cfg(not(feature = "g729"))]
            {
                Err(CodecError::InvalidCodecName)
            }
        } else if cfg!(feature = "opus") && b.eq_ignore_ascii_case(b"opus") {
            #[cfg(feature = "opus")]
            {
                Ok(CodecType::Opus)
            }
            #[cfg(not(feature = "opus"))]
            {
                Err(CodecError::InvalidCodecName)
            }
        } else if b.eq_ignore_ascii_case(b"telephone-event") {
            Ok(CodecType::TelephoneEvent)
        } else {
            Err(CodecError::InvalidCodecName)
        }
    }
}

// ----------------------------------------------------------------------------
// Byte <-> sample slice helpers
// ----------------------------------------------------------------------------

/// Write the little-endian byte representation of `samples` into `out`.
///
/// Returns the number of bytes written (`samples.len() * 2`). Returns
/// [`CodecError::BufferTooSmall`] if `out` is too small.
pub fn samples_to_bytes_into(samples: &[Sample], out: &mut [u8]) -> Result<usize, CodecError> {
    let needed = core::mem::size_of_val(samples);
    if out.len() < needed {
        return Err(CodecError::BufferTooSmall);
    }
    if samples.is_empty() {
        return Ok(0);
    }
    // A `&mut [u8]` is only guaranteed 1-byte aligned (and an empty buffer's
    // dangling pointer is 1-aligned), so the i16 view is only sound when the
    // pointer happens to be properly aligned; otherwise fall back to a
    // bytewise copy.
    let dst_aligned = out
        .as_mut_ptr()
        .addr()
        .is_multiple_of(core::mem::align_of::<Sample>());
    #[cfg(target_endian = "little")]
    if dst_aligned {
        // SAFETY: `out` is large enough (checked above) and its pointer is
        // aligned to `Sample`; `[u8; N*2]` and `[i16; N]` then have the same
        // layout.
        let dst = unsafe {
            core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut Sample, samples.len())
        };
        dst.copy_from_slice(samples);
        return Ok(needed);
    }
    for (i, s) in samples.iter().enumerate() {
        let b = s.to_le_bytes();
        out[2 * i] = b[0];
        out[2 * i + 1] = b[1];
    }
    Ok(needed)
}

/// Decode the little-endian bytes into `out` as `Sample` (i16) values.
///
/// Returns the number of samples written (`u8_data.len() / 2`). Returns
/// [`CodecError::BufferTooSmall`] if `out` cannot hold them all.
pub fn bytes_to_samples_into(u8_data: &[u8], out: &mut [Sample]) -> Result<usize, CodecError> {
    let n = u8_data.len() / core::mem::size_of::<Sample>();
    if out.len() < n {
        return Err(CodecError::BufferTooSmall);
    }
    if n == 0 {
        return Ok(0);
    }
    // See `samples_to_bytes_into`: a `&[u8]` is only guaranteed 1-byte
    // aligned, so only build the i16 view when the pointer is aligned.
    let src_aligned = u8_data
        .as_ptr()
        .addr()
        .is_multiple_of(core::mem::align_of::<Sample>());
    #[cfg(target_endian = "little")]
    if src_aligned {
        // SAFETY: see `samples_to_bytes_into`.
        let src = unsafe { core::slice::from_raw_parts(u8_data.as_ptr() as *const Sample, n) };
        out[..n].copy_from_slice(src);
        return Ok(n);
    }
    for (i, chunk) in u8_data.chunks_exact(2).enumerate() {
        out[i] = (chunk[0] as i16) | ((chunk[1] as i16) << 8);
    }
    Ok(n)
}

#[cfg(feature = "std")]
pub fn samples_to_bytes(samples: &[Sample]) -> Vec<u8> {
    let mut out = vec![0u8; core::mem::size_of_val(samples)];
    let _ = samples_to_bytes_into(samples, &mut out);
    out
}

#[cfg(feature = "std")]
pub fn bytes_to_samples(u8_data: &[u8]) -> PcmBuf {
    let n = u8_data.len() / core::mem::size_of::<Sample>();
    let mut out = vec![0i16; n];
    let _ = bytes_to_samples_into(u8_data, &mut out);
    out
}
