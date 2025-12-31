use audio_codec::resample;
use clap::Parser;
use hound::{WavReader, WavSpec, WavWriter};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input WAV file
    #[arg(short, long)]
    input: PathBuf,

    /// Output WAV file
    #[arg(short, long)]
    output: PathBuf,

    /// Target sample rate
    #[arg(short, long, default_value_t = 8000)]
    rate: u32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Reading from: {:?}", args.input);
    let mut reader = WavReader::open(&args.input)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        anyhow::bail!("Only mono WAV files are supported for this example");
    }

    if spec.sample_format != hound::SampleFormat::Int {
        anyhow::bail!("Only integer sample format is supported");
    }

    if spec.bits_per_sample != 16 {
        anyhow::bail!("Only 16-bit WAV files are supported");
    }

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    println!(
        "Original: {} samples, {} Hz",
        samples.len(),
        spec.sample_rate
    );

    let start = std::time::Instant::now();
    let resampled = resample(&samples, spec.sample_rate, args.rate);
    let duration = start.elapsed();

    println!(
        "Resampled: {} samples, {} Hz (took {:?})",
        resampled.len(),
        args.rate,
        duration
    );

    let out_spec = WavSpec {
        channels: 1,
        sample_rate: args.rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(&args.output, out_spec)?;
    for sample in resampled {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    println!("Saved to: {:?}", args.output);

    Ok(())
}
