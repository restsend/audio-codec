use std::hint::black_box;

use audio_codec::{CodecType, create_decoder, create_encoder, resampler, resampler_fixed};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_codec(c: &mut Criterion) {
    let codecs = [
        CodecType::PCMU,
        CodecType::PCMA,
        CodecType::G722,
        #[cfg(feature = "g729")]
        CodecType::G729,
        #[cfg(feature = "opus")]
        CodecType::Opus,
        CodecType::TelephoneEvent,
    ];

    for codec in codecs {
        let name = format!("{:?}", codec);
        let mut group = c.benchmark_group(&name);

        let mut encoder = create_encoder(codec);
        let mut decoder = create_decoder(codec);

        let sample_rate = encoder.sample_rate();
        let channels = encoder.channels();

        // Use 20ms of audio for each codec
        let samples_count = (sample_rate as f64 * 0.02) as usize * channels as usize;
        let pcm_samples = vec![0i16; samples_count];

        // Warm up / Get encoded data for decoder benchmark
        let encoded_data = encoder.encode(&pcm_samples);

        group.bench_function("encode_20ms", |b| {
            b.iter(|| encoder.encode(black_box(&pcm_samples)))
        });

        group.bench_function("decode_20ms", |b| {
            b.iter(|| decoder.decode(black_box(&encoded_data)))
        });

        group.finish();
    }
}

fn bench_resampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("Resampler");
    for (from, to) in [
        (48000, 8000),
        (8000, 48000),
        (16000, 8000),
        (8000, 16000),
        (16000, 48000),
        (44100, 48000),
    ] {
        // 20ms of a sine tone
        let input: Vec<i16> = (0..from / 50)
            .map(|i| ((i as f32 * 0.05).sin() * 8000.0) as i16)
            .collect();

        let mut f32_core = resampler::BoxedResampler::new(from, to).unwrap();
        let mut out = vec![0i16; f32_core.max_output_samples(input.len())];
        group.bench_function(format!("f32_{from}_to_{to}_20ms"), |b| {
            b.iter(|| f32_core.resample_into(black_box(&input), &mut out))
        });

        let mut q15_core = resampler_fixed::BoxedResampler::new(from, to).unwrap();
        let mut out = vec![0i16; q15_core.max_output_samples(input.len())];
        group.bench_function(format!("q15_{from}_to_{to}_20ms"), |b| {
            b.iter(|| q15_core.resample_into(black_box(&input), &mut out))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_codec, bench_resampler);
criterion_main!(benches);
