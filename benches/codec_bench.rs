use audio_codec::{CodecType, create_decoder, create_encoder};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_codec(c: &mut Criterion) {
    let codecs = [
        CodecType::PCMU,
        CodecType::PCMA,
        CodecType::G722,
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

criterion_group!(benches, bench_codec);
criterion_main!(benches);
