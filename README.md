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
use audio_codec::g722::G722Encoder;
use audio_codec::Encoder;

fn main() {
    let mut encoder = G722Encoder::new();
    let pcm_samples: Vec<i16> = vec![0; 320]; // 16kHz mono
    let encoded_data = encoder.encode(&pcm_samples);
    
    println!("Encoded into {} bytes", encoded_data.len());
}
```

## License

This project is licensed under the MIT License.
