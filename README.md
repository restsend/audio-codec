# audio-codec

A collection of VoIP audio codecs implemented in or wrapped for Rust. This crate provides a unified interface for encoding and decoding various audio formats commonly used in SIP, VoIP, and WebRTC applications.

## Supported Codecs

| Codec | Implementation | Feature |
|-------|----------------|---------|
| **G.711 (PCMA/PCMU)** | Pure Rust | Built-in |
| **G.722** | Pure Rust | Built-in |
| **G.729** | Wrapper (`g729-sys`) | Built-in |
| **Opus** | Wrapper (`opusic-sys`) | `opus` (default) |
| **Telephone Event** | RFC 4733 | Built-in |

## Features

- **Unified API**: Simple `Encoder` and `Decoder` traits for all codecs.
- **Resampler**: Built-in audio resampling utility.
- **Lightweight**: Minimal dependencies for core codecs.

## Performance

Measured on Apple M2 Pro (processing **20ms** audio frames):

| Codec | Encode (20ms) | Decode (20ms) | Rate |
|-------|---------------|---------------|------|
| **PCMU/A** | ~50 ns | ~80 ns | 8kHz |
| **G.722** | ~6.1 µs | ~3.7 µs | 16kHz |
| **G.729** | ~21.8 µs | ~6.2 µs | 8kHz |
| **Opus** | ~84.1 µs | ~21.8 µs | 48kHz |

*Note: Benchmarks run using `cargo bench`. Performance may vary by hardware and configuration.*

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

## License

This project is licensed under the MIT License.
