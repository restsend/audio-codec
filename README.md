# audio-codec

A collection of VoIP audio codecs implemented for Rust. This crate provides a unified interface for encoding and decoding various audio formats commonly used in SIP, VoIP, and WebRTC applications.

## Supported Codecs

| Codec | Implementation | Feature | `no_std` |
|-------|----------------|---------|----------|
| **G.711 (PCMA/PCMU)** | Pure Rust | Built-in | yes (heap-free) |
| **G.722** | Pure Rust | Built-in | yes (heap-free) |
| **G.729** | Pure Rust (`g729-sys`) | Built-in | yes (heap-free) |
| **Opus** | Pure Rust (`opus-rs`) | `opus` (on by default) | yes (needs `alloc`) |
| **Telephone Event** | RFC 4733 | Built-in | yes (heap-free) |
| **Resampler** | Polyphase FIR | Built-in | yes (heap-free) |

## Features

- **Unified API**: Simple `Encoder` and `Decoder` traits for all codecs.
- **`no_std` support**: every codec runs without `std`. G.711/G.722/G.729, the
  resampler and telephone-event are fully **heap-free** (no allocator needed);
  Opus needs an `alloc` (its internal working set is ~250 KB and is boxed).
- **Resampler**: Built-in audio resampling utility.

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
audio-codec = "0.4"   # Opus is enabled by default
```

To opt out of Opus (smaller build, no `alloc` needed):

```toml
[dependencies]
audio-codec = { version = "0.4", default-features = false, features = ["std"] }
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

## `no_std` support

The crate works on bare-metal targets with **no `std`**. G.711/G.722/G.729, the
resampler and telephone-event are additionally **heap-free** (no allocator);
Opus needs an `alloc` because its encoder/decoder working set (~250 KB) is
boxed. Disable the default `std` feature and use the slice-based `*_into` API:

```toml
[dependencies.audio-codec]
version = "0.4"
default-features = false
```

```rust
use audio_codec::{Encoder, Decoder, CodecError, g722};

fn g722_roundtrip(
    encoder: &mut g722::G722Encoder,
    decoder: &mut g722::G722Decoder,
    pcm: &[i16],
) -> Result<usize, CodecError> {
    // Caller-provided scratch buffers — no allocation.
    let mut enc_buf = [0u8; 160];           // 20ms @ 16kHz = 160 bytes
    let mut dec_buf = [0i16; 320];

    let n = encoder.encode_into(pcm, &mut enc_buf)?;
    let m = decoder.decode_into(&enc_buf[..n], &mut dec_buf)?;
    Ok(m)
}
```

Use `max_encode_bytes` / `max_decode_samples` to size the buffers correctly:

```rust
let needed_bytes = encoder.max_encode_bytes(pcm.len());
let needed_samples = decoder.max_decode_samples(n_bytes);
```

### Resampler in `no_std`

`Resampler` borrows a caller-provided coefficient buffer (~24 KB):

```rust
use audio_codec::resampler::{Resampler, COEFFS_LEN};

let mut coeffs: [f32; COEFFS_LEN] = [0.0; COEFFS_LEN];
let mut r = Resampler::new(8000, 16000, &mut coeffs)?;

let mut out = [0i16; 160];
let n = r.resample_into(&input[..80], &mut out)?;
```

### Opus in `no_std`

Opus also works without `std` (it needs an `alloc` — the encoder/decoder are
boxed). Use the slice-based `*_into` API exactly like the other codecs:

```rust
use audio_codec::{Encoder, Decoder, opus::{OpusEncoder, OpusDecoder}};

// 48 kHz, mono. The structs are boxed internally, so this only needs `alloc`.
let mut encoder = OpusEncoder::new(48_000, 1);
let mut decoder = OpusDecoder::new(48_000, 1);

let pcm: &[i16] = /* 20 ms @ 48 kHz = 960 mono samples */;
let mut packet = [0u8; 1275];
let n = encoder.encode_into(pcm, &mut packet)?;

let mut out = [0i16; 960];
let m = decoder.decode_into(&packet[..n], &mut out)?;
```

### Supported codecs in `no_std`

| Codec | Status |
|-------|--------|
| G.711 (PCMA/PCMU), G.722, G.729, Telephone Event | ✅ fully heap-free (no allocator) |
| Resampler | ✅ fully heap-free (borrowed coeffs buffer) |
| Opus | ✅ supported (needs `alloc`; encoder/decoder are boxed) |

The crate is verified to compile on `thumbv7em-none-eabi` and other bare-metal
targets. Float math (resampler + Opus) goes through `libm` when `std` is off.

## License

This project is licensed under the MIT License.
