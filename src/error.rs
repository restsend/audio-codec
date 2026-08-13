/// Errors returned by codec `encode_into` / `decode_into` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// The caller-provided output buffer was too small for the input.
    BufferTooSmall,
    /// The input data was malformed or had an unsupported length.
    InvalidInput,
    /// The numeric codec payload type is unknown / unsupported.
    InvalidCodecType,
    /// The codec name string did not match any known codec.
    InvalidCodecName,
    /// Codec initialisation failed (e.g. underlying library error).
    InitFailed,
    /// Decoding failed for an internal reason (e.g. corrupt bitstream).
    DecodeFailed,
    /// Encoding failed for an internal reason.
    EncodeFailed,
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            CodecError::BufferTooSmall => "output buffer too small",
            CodecError::InvalidInput => "invalid input data",
            CodecError::InvalidCodecType => "invalid codec payload type",
            CodecError::InvalidCodecName => "invalid codec name",
            CodecError::InitFailed => "codec initialization failed",
            CodecError::DecodeFailed => "decode failed",
            CodecError::EncodeFailed => "encode failed",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CodecError {}
