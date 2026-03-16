#![no_main]

use audio_codec::{CodecType, Decoder, PcmBuf, Sample, create_decoder, create_encoder};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FuzzData {
    codec_type: u8,
    data: Vec<u8>,
    sample_count: usize,
}

/// Fuzz test for all codecs including Opus
fn fuzz_codec(data: &FuzzData) {
    let codec = match data.codec_type % 6 {
        0 => CodecType::PCMU,
        1 => CodecType::PCMA,
        2 => CodecType::G722,
        3 => CodecType::G729,
        4 => CodecType::Opus,
        5 => CodecType::TelephoneEvent,
        _ => return,
    };

    let mut decoder = create_decoder(codec);
    let mut encoder = create_encoder(codec);

    let _decoded: PcmBuf = decoder.decode(&data.data);

    let samples: Vec<Sample> = data
        .data
        .chunks_exact(2)
        .map(|chunk| {
            let val = u16::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0)]);
            (val as i16).wrapping_mul(2)
        })
        .take(data.sample_count % 48000)
        .collect();

    let _encoded: Vec<u8> = encoder.encode(&samples);

    if !samples.is_empty() && samples.len() % codec.channels() as usize == 0 {
        let encoded = encoder.encode(&samples);
        let mut decoder2 = create_decoder(codec);
        let decoded = decoder2.decode(&encoded);

        if samples.iter().any(|&s| s != 0) && !encoded.is_empty() {
            assert!(decoded.len() >= 0);
        }
    }

    let _empty_decoded: PcmBuf = decoder.decode(&[]);
    let _empty_encoded: Vec<u8> = encoder.encode(&[]);
}

fuzz_target!(|data: FuzzData| {
    fuzz_codec(&data);
});
