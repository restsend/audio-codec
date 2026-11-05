//! `Decoder::conceal_into` / `conceal` across codecs.
#![cfg(feature = "std")]

use audio_codec::{CodecType, create_decoder, create_encoder};

/// A frame of a continuous 150 Hz tone.
fn tone(rate: u32, frame_len: usize, frame: usize) -> Vec<i16> {
    (0..frame_len)
        .map(|i| {
            let t = (frame * frame_len + i) as f32 / rate as f32;
            (8000.0 * (2.0 * std::f32::consts::PI * 150.0 * t).sin()) as i16
        })
        .collect()
}

#[cfg(any(feature = "plc", feature = "g729"))]
fn rms(s: &[i16]) -> f64 {
    (s.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
}

#[cfg(feature = "plc")]
fn max_step(s: &[i16]) -> i32 {
    s.windows(2)
        .map(|w| (w[1] as i32 - w[0] as i32).abs())
        .max()
        .unwrap_or(0)
}

/// Decode a tone, lose one frame, conceal it, decode on: the concealment
/// continues the tone and the audio around it has no step.
#[cfg(feature = "plc")]
fn conceals_a_tone(codec: CodecType, rate: u32, frame_len: usize) {
    let mut enc = create_encoder(codec);
    let mut dec = create_decoder(codec);
    assert!(dec.enable_concealment(), "{codec:?}");
    let mut played = vec![];
    for f in 0..8 {
        played.extend(dec.decode(&enc.encode(&tone(rate, frame_len, f))));
    }
    // frame 8 is lost
    enc.encode(&tone(rate, frame_len, 8));
    let concealed = dec.conceal(frame_len);
    assert_eq!(concealed.len(), frame_len, "{codec:?}");
    assert!(
        rms(&concealed) > 0.5 * rms(&tone(rate, frame_len, 8)),
        "{codec:?} concealment too quiet: rms {}",
        rms(&concealed)
    );
    played.extend(&concealed);
    played.extend(dec.decode(&enc.encode(&tone(rate, frame_len, 9))));
    // the tone's own steepest step is ~470 at 16 kHz, ~940 at 8 kHz; a gap
    // or a misaligned seam would jump by thousands
    let around = &played[played.len() - 3 * frame_len..];
    assert!(
        max_step(around) < 2500,
        "{codec:?} step {} around the loss",
        max_step(around)
    );
}

#[cfg(feature = "plc")]
#[test]
fn pcma_conceals() {
    conceals_a_tone(CodecType::PCMA, 8000, 160);
}

#[cfg(feature = "plc")]
#[test]
fn pcmu_conceals() {
    conceals_a_tone(CodecType::PCMU, 8000, 160);
}

#[cfg(feature = "plc")]
#[test]
fn g722_conceals() {
    conceals_a_tone(CodecType::G722, 16000, 320);
}

#[cfg(feature = "plc")]
#[test]
fn long_loss_fades_to_silence() {
    for (codec, frame_len) in [(CodecType::PCMA, 160), (CodecType::G722, 320)] {
        let rate = if codec == CodecType::G722 {
            16000
        } else {
            8000
        };
        let mut enc = create_encoder(codec);
        let mut dec = create_decoder(codec);
        dec.enable_concealment();
        for f in 0..8 {
            dec.decode(&enc.encode(&tone(rate, frame_len, f)));
        }
        let frames: Vec<_> = (0..4).map(|_| dec.conceal(frame_len)).collect();
        assert!(rms(&frames[1]) < rms(&frames[0]), "{codec:?}");
        assert!(
            frames[3].iter().all(|s| *s == 0),
            "{codec:?} not silent at 60 ms"
        );
    }
}

#[cfg(feature = "plc")]
#[test]
fn concealment_before_any_audio_is_silence() {
    // a stream that starts with a loss: silence until there is audio to
    // conceal from, on every call (the history fills with that silence)
    for (codec, frame_len) in [(CodecType::PCMA, 160), (CodecType::G722, 320)] {
        let mut dec = create_decoder(codec);
        dec.enable_concealment();
        for _ in 0..6 {
            let out = dec.conceal(frame_len);
            assert_eq!(out.len(), frame_len, "{codec:?}");
            assert!(out.iter().all(|s| *s == 0), "{codec:?}");
        }
    }
}

#[cfg(feature = "plc")]
#[test]
fn empty_packet_does_not_end_a_loss() {
    let mut enc = create_encoder(CodecType::PCMA);
    let mut dec = create_decoder(CodecType::PCMA);
    dec.enable_concealment();
    for f in 0..8 {
        dec.decode(&enc.encode(&tone(8000, 160, f)));
    }
    let first = dec.conceal(160);
    let second = dec.conceal(160);
    assert!(dec.decode(&[]).is_empty());
    // carries on fading rather than restarting at full level
    let third = dec.conceal(160);
    assert!(rms(&second) < rms(&first));
    assert!(
        rms(&third) < rms(&second),
        "loss restarted after an empty packet"
    );
}

#[cfg(feature = "plc")]
#[test]
fn g722_recovers_after_a_long_loss() {
    let mut enc = create_encoder(CodecType::G722);
    let mut dec = create_decoder(CodecType::G722);
    dec.enable_concealment();
    for f in 0..8 {
        dec.decode(&enc.encode(&tone(16000, 320, f)));
    }
    // 200 ms lost: faded out long before it ends
    for f in 8..18 {
        enc.encode(&tone(16000, 320, f));
        assert_eq!(dec.conceal(320).len(), 320);
    }
    // the tone comes back at its level within a few packets
    let mut last = vec![];
    for f in 18..24 {
        last = dec.decode(&enc.encode(&tone(16000, 320, f)));
    }
    assert!(
        rms(&last) > 0.7 * rms(&tone(16000, 320, 23)),
        "rms {} after recovery",
        rms(&last)
    );
}

#[test]
fn decoders_without_concealment_write_nothing() {
    let mut dec = create_decoder(CodecType::TelephoneEvent);
    let mut out = [1i16; 80];
    assert_eq!(dec.conceal_into(80, &mut out), Ok(0));
    assert!(dec.conceal(80).is_empty());
}

/// Voiced, speech-like 8 kHz: 8 harmonics of a pitch gliding 120 → 180 Hz
/// under a slow envelope, in 10 ms frames.
#[cfg(feature = "g729")]
fn voice_8k(frames: usize) -> Vec<Vec<i16>> {
    let mut phase = 0f32;
    let pcm: Vec<i16> = (0..frames * 80)
        .map(|i| {
            let t = i as f32 / 8000.0;
            let f0 = 120.0 + 60.0 * (t / 0.6).min(1.0);
            phase += 2.0 * std::f32::consts::PI * f0 / 8000.0;
            let env = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 2.0 * t).sin();
            let v: f32 = (1..=8).map(|k| (k as f32 * phase).sin() / k as f32).sum();
            (6000.0 * env * v) as i16
        })
        .collect();
    pcm.chunks(80).map(<[i16]>::to_vec).collect()
}

#[cfg(feature = "g729")]
#[test]
fn g729_conceals_with_its_own_erasure_concealment_faded() {
    let mut enc = create_encoder(CodecType::G729);
    let mut dec = create_decoder(CodecType::G729);
    assert!(dec.enable_concealment());
    for frame in voice_8k(30) {
        dec.decode(&enc.encode(&frame));
    }
    // 80 ms lost, 10 ms at a time
    let levels: Vec<f64> = (0..8).map(|_| rms(&dec.conceal(80))).collect();
    assert!(levels[0] > 500.0, "concealment is silent: {levels:?}");
    assert!(
        levels[1] < levels[0] && levels[4] < levels[1],
        "not fading: {levels:?}"
    );
    assert!(
        levels[6..].iter().all(|l| *l == 0.0),
        "not silent at 60 ms: {levels:?}"
    );
    // a 20 ms packet at once
    assert_eq!(dec.conceal(160).len(), 160);
}

/// Concealing keeps G.729's predictors in step with the encoder's: the
/// packets after a loss decode closer to a lossless decode than when the
/// loss is skipped.
#[cfg(feature = "g729")]
#[test]
fn g729_concealment_keeps_the_decoder_in_step() {
    let mut enc = create_encoder(CodecType::G729);
    let packets: Vec<Vec<u8>> = voice_8k(80).iter().map(|f| enc.encode(f)).collect();
    let error_db = |conceal: bool| {
        let (mut err, mut sig) = (0f64, 0f64);
        for lost in (20..70).step_by(10) {
            let mut clean = create_decoder(CodecType::G729);
            let mut dec = create_decoder(CodecType::G729);
            for p in &packets[..lost] {
                clean.decode(p);
                dec.decode(p);
            }
            for p in &packets[lost..lost + 2] {
                clean.decode(p);
                if conceal {
                    dec.conceal(80);
                }
            }
            for p in &packets[lost + 2..lost + 7] {
                let (a, b) = (dec.decode(p), clean.decode(p));
                err += a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                    .sum::<f64>();
                sig += b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>();
            }
        }
        10.0 * (err / sig).log10()
    };
    let (concealed, skipped) = (error_db(true), error_db(false));
    println!("after a 20 ms loss: concealed {concealed:.1} dB, skipped {skipped:.1} dB");
    // measured: concealed 0.6 dB, skipped 3.1 dB on this signal; on TTS
    // speech, -3.7 dB and +5.0 dB
    assert!(
        concealed < skipped - 1.5,
        "concealing no better than skipping: {concealed:.1} vs {skipped:.1} dB"
    );
}

#[cfg(feature = "g729")]
#[test]
fn g729_sid_frames_decode_to_comfort_noise() {
    use audio_codec::g729::{G729Decoder, G729Encoder};
    use audio_codec::{Decoder, Encoder};
    let mut enc = G729Encoder::with_vad(true);
    let mut dec = G729Decoder::new();
    let mut speech = 0f64;
    let mut sid = None;
    for (i, frame) in voice_8k(20)
        .into_iter()
        .chain((0..30).map(|_| vec![0i16; 80]))
        .enumerate()
    {
        let payload = enc.encode(&frame);
        let pcm = dec.decode(&payload);
        if i < 20 {
            speech = speech.max(rms(&pcm));
        }
        if payload.len() == 2 {
            sid = Some(pcm);
        }
    }
    let sid = sid.expect("no SID frame");
    assert_eq!(sid.len(), 80, "a SID frame is 10 ms of comfort noise");
    assert!(
        rms(&sid) < speech / 10.0,
        "{} vs speech {speech}",
        rms(&sid)
    );
    assert_eq!(dec.max_decode_samples(2), 80);
    assert_eq!(dec.max_decode_samples(12), 160);
    assert_eq!(dec.max_decode_samples(20), 160);
}

#[cfg(feature = "g729")]
#[test]
fn g729_conceals_10_ms_at_a_time_and_from_nothing() {
    let mut dec = create_decoder(CodecType::G729);
    let mut out = [0i16; 160];
    assert_eq!(
        dec.conceal_into(100, &mut out),
        Err(audio_codec::CodecError::InvalidInput)
    );
    assert_eq!(
        dec.conceal_into(160, &mut out[..80]),
        Err(audio_codec::CodecError::BufferTooSmall)
    );
    // before any audio: whatever the decoder's initial state gives, faded
    assert_eq!(dec.conceal_into(160, &mut out), Ok(160));
}

/// Without enabling, or without the `plc` feature, G.711 and G.722 leave
/// concealment to the caller.
#[test]
fn concealment_is_off_until_enabled() {
    for (codec, frame_len) in [
        (CodecType::PCMA, 160),
        (CodecType::PCMU, 160),
        (CodecType::G722, 320),
    ] {
        let rate = if codec == CodecType::G722 {
            16000
        } else {
            8000
        };
        let mut enc = create_encoder(codec);
        let mut dec = create_decoder(codec);
        dec.decode(&enc.encode(&tone(rate, frame_len, 0)));
        let mut out = vec![1i16; frame_len];
        assert_eq!(dec.conceal_into(frame_len, &mut out), Ok(0), "{codec:?}");
        assert!(
            out.iter().all(|s| *s == 1),
            "{codec:?} wrote while disabled"
        );
    }
}

#[cfg(feature = "plc")]
#[test]
fn disabled_concealment_costs_a_pointer() {
    use audio_codec::pcma::PcmaDecoder;
    use audio_codec::pcmu::PcmuDecoder;
    assert_eq!(
        std::mem::size_of::<PcmaDecoder>(),
        std::mem::size_of::<usize>()
    );
    assert_eq!(
        std::mem::size_of::<PcmuDecoder>(),
        std::mem::size_of::<usize>()
    );
}

/// Without redundancy to rebuild from, `recover` conceals.
#[cfg(feature = "plc")]
#[test]
fn recover_falls_back_to_concealment() {
    let mut enc = create_encoder(CodecType::PCMA);
    let mut dec = create_decoder(CodecType::PCMA);
    dec.enable_concealment();
    for f in 0..8 {
        dec.decode(&enc.encode(&tone(8000, 160, f)));
    }
    let next = enc.encode(&tone(8000, 160, 9));
    let recovered = dec.recover(&next, 160);
    assert_eq!(recovered.len(), 160);
    assert!(rms(&recovered) > 0.5 * rms(&tone(8000, 160, 8)));
}
