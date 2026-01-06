use audio_codec::{CodecType, create_decoder, create_encoder};

fn main() {
    // List of codecs to demonstrate
    let codec_types = [
        CodecType::PCMU,
        CodecType::PCMA,
        CodecType::G722,
        CodecType::G729,
        #[cfg(feature = "opus")]
        CodecType::Opus,
    ];

    for codec_type in codec_types {
        println!("--- Testing Codec: {:?} ---", codec_type);

        let mut encoder = create_encoder(codec_type);
        let mut decoder = create_decoder(codec_type);

        let sample_rate = encoder.sample_rate();
        let channels = encoder.channels();
        let frame_size = (sample_rate as f64 * 0.02) as usize * channels as usize;
        let pcm_input = vec![0i16; frame_size];

        println!(
            "Input:  {} samples ({}Hz, {}ch)",
            pcm_input.len(),
            sample_rate,
            channels
        );

        let start_enc = std::time::Instant::now();
        let encoded = encoder.encode(&pcm_input);
        let duration_enc = start_enc.elapsed();

        println!("Encoded: {} bytes (took {:?})", encoded.len(), duration_enc);

        let start_dec = std::time::Instant::now();
        let decoded = decoder.decode(&encoded);
        let duration_dec = start_dec.elapsed();

        println!(
            "Decoded: {} samples (took {:?})",
            decoded.len(),
            duration_dec
        );
        println!();
    }
}
