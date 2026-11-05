//! `CodecType::is_silence_frame`: telling a sender that went quiet (DTX)
//! from one whose packets were lost.

#![cfg(feature = "std")]

use audio_codec::CodecType;
#[cfg(any(feature = "g729", feature = "opus"))]
use audio_codec::Encoder;

/// 10 ms frames at `rate` of a tone, then digital silence.
#[cfg(any(feature = "g729", feature = "opus"))]
fn tone_then_silence(rate: u32, frame_ms: u32, tone: usize, silence: usize) -> Vec<Vec<i16>> {
    let n = (rate * frame_ms / 1000) as usize;
    (0..tone + silence)
        .map(|f| {
            (0..n)
                .map(|i| {
                    if f >= tone {
                        return 0;
                    }
                    let t = (f * n + i) as f32 / rate as f32;
                    (3000.0 * (2.0 * core::f32::consts::PI * 300.0 * t).sin()) as i16
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "g729")]
#[test]
fn g729_sid_frames_are_silence_frames() {
    use audio_codec::g729::G729Encoder;
    let mut enc = G729Encoder::with_vad(true);
    let payloads: Vec<Vec<u8>> = tone_then_silence(8000, 10, 20, 20)
        .iter()
        .map(|f| enc.encode(f))
        .collect();
    let codec = CodecType::G729;
    // after its hangover the encoder goes quiet with a 2-byte SID frame,
    // then sends nothing
    let at = payloads
        .iter()
        .position(|p| p.len() == 2)
        .expect("no SID frame in silence");
    let sid = &payloads[at];
    assert!(at >= 20 && payloads[at + 1].is_empty());
    assert!(codec.is_silence_frame(sid));
    assert!(payloads[..at].iter().all(|p| !codec.is_silence_frame(p)));
    // speech frames with a SID at the end of the payload (RFC 3551)
    let mut tail = payloads[0].clone();
    tail.extend_from_slice(sid);
    assert!(codec.is_silence_frame(&tail));
    let mut two = payloads[0].clone();
    two.extend_from_slice(&payloads[1]);
    assert!(!codec.is_silence_frame(&two));
}

#[cfg(feature = "opus")]
#[test]
fn opus_dtx_packets_are_silence_frames() {
    use audio_codec::opus::OpusEncoder;
    let mut enc = OpusEncoder::new(48000, 1);
    let codec = CodecType::Opus;
    for frame in tone_then_silence(48000, 20, 10, 10) {
        let packet = enc.encode(&frame);
        // coded speech or silence, not DTX: more than the TOC byte and one
        assert!(!codec.is_silence_frame(&packet), "{} bytes", packet.len());
    }
    // a DTX packet: the TOC byte alone, or with one more byte
    assert!(codec.is_silence_frame(&[0x78]));
    assert!(codec.is_silence_frame(&[0x78, 0x00]));
    assert!(!codec.is_silence_frame(&[]));
}

#[test]
fn codecs_without_dtx_have_no_silence_frames() {
    for codec in [
        CodecType::PCMU,
        CodecType::PCMA,
        CodecType::G722,
        CodecType::TelephoneEvent,
    ] {
        for len in [0, 1, 2, 4, 160] {
            assert!(!codec.is_silence_frame(&vec![0u8; len]), "{codec:?} {len}");
        }
    }
}
