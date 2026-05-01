use audio_codec::opus::OpusDecoder;
use audio_codec::Decoder;
use opusic_sys::{
    OPUS_APPLICATION_AUDIO, OPUS_OK, OPUS_SET_BITRATE_REQUEST,
    opus_encode, opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy,
};

const SAMPLE_RATE: i32 = 48000;
const FRAME_MS: usize = 20;
const FRAME_SIZE: usize = (SAMPLE_RATE as usize * FRAME_MS) / 1000;

fn encode_mono_with_opusic(pcm: &[i16]) -> Vec<u8> {
    let mut err = 0;
    let ptr = unsafe { opus_encoder_create(SAMPLE_RATE, 1, OPUS_APPLICATION_AUDIO, &mut err) };
    assert!(err == OPUS_OK && !ptr.is_null(), "create mono encoder failed");
    unsafe { let _ = opus_encoder_ctl(ptr, OPUS_SET_BITRATE_REQUEST, 48000); }

    let mut packet = vec![0u8; 1500];
    let n = unsafe {
        opus_encode(ptr, pcm.as_ptr(), FRAME_SIZE as i32, packet.as_mut_ptr(), packet.len() as i32)
    };
    assert!(n > 0, "opusic encode failed: {n}");
    packet.truncate(n as usize);

    unsafe { opus_encoder_destroy(ptr); }
    packet
}

fn encode_stereo_with_opusic(pcm: &[i16]) -> Vec<u8> {
    let mut err = 0;
    let ptr = unsafe { opus_encoder_create(SAMPLE_RATE, 2, OPUS_APPLICATION_AUDIO, &mut err) };
    assert!(err == OPUS_OK && !ptr.is_null(), "create stereo encoder failed");
    unsafe { let _ = opus_encoder_ctl(ptr, OPUS_SET_BITRATE_REQUEST, 64000); }

    let mut packet = vec![0u8; 1500];
    let n = unsafe {
        opus_encode(ptr, pcm.as_ptr(), FRAME_SIZE as i32, packet.as_mut_ptr(), packet.len() as i32)
    };
    assert!(n > 0, "opusic stereo encode failed: {n}");
    packet.truncate(n as usize);

    unsafe { opus_encoder_destroy(ptr); }
    packet
}

#[test]
fn test_mono_packet_decoded_by_stereo_decoder() {
    let pcm: Vec<i16> = (0..FRAME_SIZE)
        .map(|i| ((i as f64 * 440.0 * 2.0 * std::f64::consts::PI / SAMPLE_RATE as f64).sin() * 5000.0) as i16)
        .collect();

    let mono_packet = encode_mono_with_opusic(&pcm);
    // TOC byte: bit 2 = 0 for mono
    assert_eq!(mono_packet[0] & 0x04, 0, "expected mono TOC");

    let stereo_pcm: Vec<i16> = (0..FRAME_SIZE * 2)
        .map(|i| ((i as f64 * 440.0 * 2.0 * std::f64::consts::PI / SAMPLE_RATE as f64).sin() * 5000.0) as i16)
        .collect();
    let stereo_packet = encode_stereo_with_opusic(&stereo_pcm);
    // TOC byte: bit 2 = 1 for stereo
    assert_ne!(stereo_packet[0] & 0x04, 0, "expected stereo TOC");

    // Mono packet → stereo decoder (should NOT be empty after fix)
    let mut stereo_decoder = OpusDecoder::new_default();
    let result = stereo_decoder.decode(&mono_packet);
    assert!(!result.is_empty(),
        "Mono packet decoded by stereo decoder returned EMPTY! Channel mismatch bug.");
    assert_eq!(result.len(), FRAME_SIZE, "expected {} mono samples from stereo decoder", FRAME_SIZE);

    // Stereo packet → stereo decoder (still works)
    let mut stereo_decoder2 = OpusDecoder::new_default();
    let result2 = stereo_decoder2.decode(&stereo_packet);
    assert!(!result2.is_empty());
    assert_eq!(result2.len(), FRAME_SIZE * 2, "expected {} stereo samples", FRAME_SIZE * 2);

    // Mono packet → mono decoder (still works)
    let mut mono_decoder = OpusDecoder::new(SAMPLE_RATE as u32, 1);
    let result3 = mono_decoder.decode(&mono_packet);
    assert!(!result3.is_empty());
    assert_eq!(result3.len(), FRAME_SIZE, "expected {} mono samples", FRAME_SIZE);
}
