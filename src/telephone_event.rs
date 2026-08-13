use super::{CodecError, Decoder, Encoder, Sample};

pub struct TelephoneEventDecoder {}

impl TelephoneEventDecoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for TelephoneEventDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for TelephoneEventDecoder {
    fn decode_into(&mut self, _data: &[u8], _out: &mut [Sample]) -> Result<usize, CodecError> {
        Ok(0)
    }

    fn max_decode_samples(&self, _n_bytes: usize) -> usize {
        0
    }

    fn sample_rate(&self) -> u32 {
        8000
    }

    fn channels(&self) -> u16 {
        1
    }
}

pub struct TelephoneEventEncoder {}

impl TelephoneEventEncoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for TelephoneEventEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder for TelephoneEventEncoder {
    fn encode_into(
        &mut self,
        _samples: &[Sample],
        _out: &mut [u8],
    ) -> Result<usize, CodecError> {
        Ok(0)
    }

    fn max_encode_bytes(&self, _n_samples: usize) -> usize {
        0
    }

    fn sample_rate(&self) -> u32 {
        8000
    }

    fn channels(&self) -> u16 {
        1
    }
}
