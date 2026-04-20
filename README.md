# audio-codec

A collection of VoIP audio codecs implemented for Rust. This crate provides a unified interface for encoding and decoding various audio formats commonly used in SIP, VoIP, and WebRTC applications.

## Supported Codecs

| Codec | Implementation | Feature |
|-------|----------------|---------|
| **G.711 (PCMA/PCMU)** | Pure Rust | Built-in |
| **G.722** | Pure Rust | Built-in |
| **G.729** | Pure Rust (`g729-sys`) | Built-in |
| **Opus** | Pure Rust (`opus-rs`) | `opus` (default) |
| **Telephone Event** | RFC 4733 | Built-in |

## Features

- **Unified API**: Simple `Encoder` and `Decoder` traits for all codecs.
- **Resampler**: Built-in audio resampling utility.
- **Lightweight**: Minimal dependencies for core codecs.

## Performance

Measured on Apple M2 Pro (processing **20ms** audio frames):

| Codec | Encode (20ms) | Decode (20ms) | Rate |
|-------|---------------|---------------|------|
| **PCMU** | ~50.09 ns | ~59.73 ns | 8kHz |
| **PCMA** | ~50.23 ns | ~59.63 ns | 8kHz |
| **G.722** | ~5.02 µs | ~3.82 µs | 16kHz |
| **G.729** | ~20.50 µs | ~6.16 µs | 8kHz |
| **Opus** | ~52.34 µs | ~23.19 µs | 48kHz |

*Note: Benchmarks were run with `cargo bench --bench codec_bench` (Criterion). `create_encoder(CodecType::Opus)` currently uses the default Opus profile: 48kHz, stereo, `Application::Audio`, bitrate 64kbps, complexity 5.*

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
audio-codec = "0.3"
```

### Example: Decoding PCMA

```rust
use audio_codec::{create_decoder, CodecType};

fn main() {
    let mut decoder = create_decoder(CodecType::PCMA);
    let encoded_data: Vec<u8> = vec![/* ... */];
    let pcm_samples = decoder.decode(&encoded_data);
    
    println!("Decoded {} samples", pcm_samples.len());
}
```

### Example: Encoding G.722

```rust
use audio_codec::{create_encoder, CodecType, Encoder};

fn main() {
    let mut encoder = create_encoder(CodecType::G722);
    let pcm_samples: Vec<i16> = vec![0; 320]; // 20ms @ 16kHz mono
    let encoded_data = encoder.encode(&pcm_samples);
    
    println!("Encoded into {} bytes", encoded_data.len());
}
```

### Example: Configuring Opus (Factory API)

```rust
use audio_codec::{
    create_opus_decoder, create_opus_encoder, Decoder, Encoder,
    opus::OpusApplication,
};

fn main() {
    // Explicit Opus encoder/decoder creation
    // 48kHz stereo, Audio mode
    let mut encoder = create_opus_encoder(48_000, 2, OpusApplication::Audio);
    let mut decoder = create_opus_decoder(48_000, 2);

    let pcm_samples: Vec<i16> = vec![0; 960 * 2]; // 20ms @ 48kHz, interleaved stereo
    let encoded_data = encoder.encode(&pcm_samples);
    let decoded_pcm = decoder.decode(&encoded_data);

    println!("Encoded into {} bytes", encoded_data.len());
    println!("Decoded {} samples", decoded_pcm.len());
}
```

## License

This project is licensed under the MIT License.
