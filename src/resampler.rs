use core::f64::consts::PI as PI_F64;

use super::{CodecError, Sample};

#[cfg(feature = "std")]
use super::PcmBuf;

/// Number of polyphase filter phases.
pub const NUM_PHASES: usize = 256;
/// Number of filter taps per phase.
pub const TAPS_PER_PHASE: usize = 24;
/// Required length of the caller-provided coefficient buffer (`NUM_PHASES * TAPS_PER_PHASE`).
pub const COEFFS_LEN: usize = NUM_PHASES * TAPS_PER_PHASE;

/// Fixed-point scale for the polyphase coefficients and the FIR accumulator
/// (Q15: one unit = 1/32768).
const Q15_SHIFT: u32 = 15;

/// `f64::sin` polyfill that works in both std and no_std (via libm).
#[inline]
fn fsin(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::sin(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sin(x)
    }
}

/// `f64::sqrt` polyfill.
#[inline]
fn fsqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        f64::sqrt(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

/// `f64::round` polyfill (round half away from zero).
#[inline]
fn fround(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.round()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::round(x)
    }
}

/// Polyphase resampler.
///
/// The filter runs entirely in **fixed point** (Q15 i16 coefficients, i16
/// history, i64 accumulator, integer phase accumulator). This matters on
/// targets without an FPU (e.g. ESP32-S3 / Xtensa LX7): a soft-float `f32`
/// FIR is ~20× slower than the equivalent integer one there. Coefficients are
/// still *designed* in `f64` at construction time and quantised once.
///
/// The coefficient buffer is **Q15 `i16`** (6144 entries, 12 KB) supplied by
/// the caller so this type stays `no_std`/allocation-free.
pub struct Resampler<'a> {
    input_rate: usize,
    output_rate: usize,
    coeffs: &'a mut [i16],
    taps_per_phase: usize,
    /// Sliding window of the last `TAPS_PER_PHASE` inputs, held in a double
    /// buffer to avoid shifting on every input sample: new samples are
    /// appended at `w`, the window is `history[w - TAPS .. w]`, and the
    /// buffer is compacted once every `TAPS_PER_PHASE` inputs.
    history: [i16; 2 * TAPS_PER_PHASE],
    /// Write cursor into `history` (`TAPS_PER_PHASE ..= 2 * TAPS_PER_PHASE`).
    w: usize,
    /// Ticks from the newest input sample to the next output sample. It can
    /// exceed one input sample when downsampling.
    ///
    /// Timing runs on an integer grid of `input_rate * output_rate` ticks per
    /// second: one input sample is `output_rate` ticks and one output sample
    /// is `input_rate` ticks, so the timing is exact and never rounded.
    pos: usize,
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let x_sq = x * x * 0.25;

    for m in 1..=30 {
        term *= x_sq / (m * m) as f64;
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

fn kaiser_window(n: usize, n_total: usize, beta: f64) -> f64 {
    if n_total <= 1 {
        return 1.0;
    }
    let alpha = (n_total - 1) as f64 / 2.0;
    let x = (n as f64 - alpha) / alpha;
    let arg = beta * fsqrt(1.0 - x * x);
    bessel_i0(arg) / bessel_i0(beta)
}

impl<'a> Resampler<'a> {
    /// Create a new `Resampler`, writing Q15 polyphase filter coefficients into
    /// the caller-provided buffer.
    ///
    /// `coeffs.len()` must be at least [`COEFFS_LEN`] (= 6144 `i16`, 12 KB).
    /// The buffer is held by the resampler for its entire lifetime; it is
    /// written once here and read on every subsequent `resample_into` call.
    pub fn new(
        input_rate: usize,
        output_rate: usize,
        coeffs: &'a mut [i16],
    ) -> Result<Self, CodecError> {
        if coeffs.len() < COEFFS_LEN {
            return Err(CodecError::BufferTooSmall);
        }
        if input_rate == 0 || output_rate == 0 {
            return Err(CodecError::InvalidInput);
        }

        const KAISER_BETA: f64 = 7.0;

        let ratio = output_rate as f64 / input_rate as f64;
        let num_phases = NUM_PHASES;
        let taps_per_phase = TAPS_PER_PHASE;
        let filter_len = num_phases * taps_per_phase;

        // Integer tick periods keep the timing exact: the old Q24 step rounded
        // 1/3 and produced an extra sample on the first 16 -> 48 kHz block.
        // In the hot loop `pos < output_rate` whenever `pos + input_rate` or
        // `pos * NUM_PHASES` is computed, so neither can overflow.
        if input_rate.checked_add(output_rate).is_none()
            || output_rate.checked_mul(NUM_PHASES).is_none()
        {
            return Err(CodecError::InvalidInput);
        }

        let coeffs = &mut coeffs[..filter_len];

        let cutoff = if ratio < 1.0 {
            ratio * 0.5 * 0.95
        } else {
            0.5 * 0.95
        };

        let center = (taps_per_phase as f64 - 1.0) / 2.0;

        // Design the polyphase filter directly into the borrowed buffer.
        // We use a fixed-size scratch array on the stack (24 f64 = 192 bytes).
        let mut phase_coeffs: [f64; TAPS_PER_PHASE] = [0.0; TAPS_PER_PHASE];

        for p in 0..num_phases {
            let mut sum = 0.0_f64;

            for t in 0..taps_per_phase {
                let x = t as f64 - center - (p as f64 / num_phases as f64);

                let sinc_val = if x.abs() < 1e-10 {
                    2.0 * cutoff
                } else {
                    let x_pi = x * PI_F64;
                    fsin(x_pi * 2.0 * cutoff) / x_pi
                };

                let full_filter_idx = t * num_phases + p;
                let window = kaiser_window(full_filter_idx, filter_len, KAISER_BETA);

                phase_coeffs[t] = sinc_val * window;
                sum += phase_coeffs[t];
            }

            for t in 0..taps_per_phase {
                // Quantise the normalised tap to Q15.
                let q = fround((phase_coeffs[t] / sum) * 32768.0);
                coeffs[p * taps_per_phase + t] = q.clamp(-32768.0, 32767.0) as i16;
            }
        }

        Ok(Self {
            input_rate,
            output_rate,
            coeffs,
            taps_per_phase,
            history: [0; 2 * TAPS_PER_PHASE],
            w: TAPS_PER_PHASE,
            pos: 0,
        })
    }

    pub fn input_rate(&self) -> usize {
        self.input_rate
    }

    pub fn output_rate(&self) -> usize {
        self.output_rate
    }

    /// Q15 dot product of the 24-tap window against one phase, accumulated in
    /// `i64` (worst-case `24 * 32768 * 32767` overflows `i32`). The `i16`×`i16`
    /// product itself fits in `i32`, so only the accumulation widens.
    ///
    /// Takes fixed-size array refs so the outer loops' slice↔array `try_into`
    /// is the only length check (the `i16` indices below are unchecked).
    #[inline(always)]
    fn dot_q15(history: &[i16; TAPS_PER_PHASE], coeffs: &[i16; TAPS_PER_PHASE]) -> i32 {
        let mut acc: i64 = 0;
        for i in 0..TAPS_PER_PHASE {
            acc += ((history[i] as i32) * (coeffs[i] as i32)) as i64;
        }
        (acc >> Q15_SHIFT) as i32
    }

    /// Resample `input` into the caller-provided `out` buffer.
    ///
    /// Returns the number of samples written. Returns
    /// [`CodecError::BufferTooSmall`] if `out` cannot hold the result; use
    /// [`Self::max_output_samples`] to size it.
    pub fn resample_into(
        &mut self,
        input: &[Sample],
        out: &mut [Sample],
    ) -> Result<usize, CodecError> {
        if self.input_rate == self.output_rate {
            if out.len() < input.len() {
                return Err(CodecError::BufferTooSmall);
            }
            out[..input.len()].copy_from_slice(input);
            return Ok(input.len());
        }

        let taps = self.taps_per_phase;
        let two_taps = 2 * taps;
        let mut written = 0usize;
        // Tick periods (see `pos`): `step` per output, `interval` per input.
        // Work on locals: a field would have to be kept up to date in memory
        // at every possible early exit, a local can stay in a register.
        let (step, interval) = (self.input_rate, self.output_rate);
        let mut pos = self.pos;

        for &sample in input {
            // Compact the double buffer only once per `taps` inputs.
            if self.w == two_taps {
                self.history.copy_within(taps..two_taps, 0);
                self.w = taps;
            }
            self.history[self.w] = sample;
            self.w += 1;

            // Emit every output that falls before the next input sample.
            if pos < interval {
                let base = self.w - taps;
                let win: &[i16; TAPS_PER_PHASE] = self.history[base..base + taps]
                    .try_into()
                    .unwrap();

                while pos < interval {
                    // Where the output falls between two input samples picks
                    // the filter phase; only this choice is quantised.
                    let phase_idx = pos * NUM_PHASES / interval;
                    let offset = phase_idx * taps;
                    let phase_coeffs: &[i16; TAPS_PER_PHASE] =
                        self.coeffs[offset..offset + taps].try_into().unwrap();
                    let out_sample = Self::dot_q15(win, phase_coeffs);

                    if written >= out.len() {
                        self.pos = pos;
                        return Err(CodecError::BufferTooSmall);
                    }
                    out[written] = out_sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    written += 1;
                    pos += step;
                }
            }
            pos -= interval;
        }
        self.pos = pos;

        Ok(written)
    }

    /// Upper bound on the number of samples `resample_into` will produce for
    /// an input of `n_input` samples.
    pub fn max_output_samples(&self, n_input: usize) -> usize {
        if self.input_rate == self.output_rate {
            return n_input;
        }
        // Outputs (`input_rate` ticks apart) that fit in `n_input` input
        // periods (`output_rate` ticks each); the most fit when `pos == 0`.
        let count = (n_input as u128 * self.output_rate as u128).div_ceil(self.input_rate as u128);
        count.min(usize::MAX as u128) as usize
    }

    /// Convenience wrapper that allocates a `Vec` and calls `resample_into`.
    #[cfg(feature = "std")]
    pub fn resample(&mut self, input: &[Sample]) -> PcmBuf {
        let max = self.max_output_samples(input.len());
        let mut out = vec![0i16; max];
        match self.resample_into(input, &mut out) {
            Ok(n) => {
                out.truncate(n);
                out
            }
            Err(_) => Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.history.fill(0);
        self.w = self.taps_per_phase;
        self.pos = 0;
    }
}

/// One-shot resampling convenience helper (allocates).
///
/// Only available with the `std` feature.
#[cfg(feature = "std")]
pub fn resample(input: &[Sample], input_sample_rate: u32, output_sample_rate: u32) -> PcmBuf {
    if input_sample_rate == output_sample_rate {
        return input.to_vec();
    }
    let mut coeffs = vec![0i16; COEFFS_LEN];
    let mut r = match Resampler::new(
        input_sample_rate as usize,
        output_sample_rate as usize,
        &mut coeffs,
    ) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    r.resample(input)
}

/// Self-contained [`Resampler`] that owns its coefficient buffer (std only).
///
/// [`Resampler`] borrows a ~12 KB caller-provided coefficient slice, which
/// makes it awkward to store as a long-lived struct field (the buffer must
/// outlive the resampler). `BoxedResampler` heap-allocates the coefficients
/// once and keeps them alive for the resampler's whole lifetime, restoring
/// the pre-0.4 ergonomic `new(input_rate, output_rate)` + `resample(..)`
/// call shape for std users that hold resamplers in struct fields.
#[cfg(feature = "std")]
pub struct BoxedResampler {
    /// Heap allocation: the buffer address is stable for the box's lifetime
    /// and never reallocated, which is what the `Resampler<'static>` borrow
    /// below relies on. Field order matters for drop: `inner` (borrower)
    /// must drop before `coeffs` (borrowed).
    inner: Resampler<'static>,
    /// Kept alive only to back `inner`'s `'static` borrow; never read.
    #[allow(dead_code)]
    coeffs: Box<[i16]>,
}

#[cfg(feature = "std")]
impl BoxedResampler {
    /// Create a resampler that owns its polyphase filter coefficients.
    ///
    /// Fails only for zero rates (the coefficient buffer is always sized
    /// correctly internally).
    pub fn new(input_rate: usize, output_rate: usize) -> Result<Self, CodecError> {
        let mut coeffs = vec![0i16; COEFFS_LEN].into_boxed_slice();
        // SAFETY: `coeffs` is a heap box whose address cannot change while
        // the allocation lives. We extend the borrow to 'static solely to
        // store both the buffer and its borrower in the same struct; the
        // struct's field order drops `inner` before `coeffs`, and nothing
        // else can reach the buffer (it is moved into `Self` right after).
        let inner = unsafe {
            let ptr = coeffs.as_mut_ptr();
            let loan: &'static mut [i16] = core::slice::from_raw_parts_mut(ptr, coeffs.len());
            Resampler::new(input_rate, output_rate, loan)?
        };
        Ok(Self { inner, coeffs })
    }

    /// Convenience wrapper that allocates the output buffer.
    pub fn resample(&mut self, input: &[Sample]) -> PcmBuf {
        self.inner.resample(input)
    }

    /// Resample into a caller-provided buffer (no allocation).
    pub fn resample_into(
        &mut self,
        input: &[Sample],
        out: &mut [Sample],
    ) -> Result<usize, CodecError> {
        self.inner.resample_into(input, out)
    }

    /// Upper bound on the output sample count for `n_input` input samples.
    pub fn max_output_samples(&self, n_input: usize) -> usize {
        self.inner.max_output_samples(n_input)
    }

    /// Reset the resampling history (e.g. after a source discontinuity).
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;
    use std::time::Instant;

    fn new_resampler(input_rate: usize, output_rate: usize) -> Resampler<'static> {
        // Leak intentionally: tests are short-lived and we need a 'static reference.
        let coeffs: &'static mut [i16] = Box::leak(vec![0i16; COEFFS_LEN].into_boxed_slice());
        Resampler::new(input_rate, output_rate, coeffs).expect("resampler init")
    }

    #[test]
    fn test_exact_sample_counts_for_repeated_20ms_blocks() {
        for input_rate in [8000, 16000, 44100, 48000] {
            for output_rate in [8000, 16000, 44100, 48000] {
                let mut r = new_resampler(input_rate, output_rate);
                let input = vec![1000; input_rate / 50];
                let mut output = vec![0; output_rate / 50];
                for block in 0..10 {
                    let n = r.resample_into(&input, &mut output).unwrap_or_else(|e| {
                        panic!("{input_rate} -> {output_rate}, block {block}: {e:?}")
                    });
                    assert_eq!(n, output.len(), "{input_rate} -> {output_rate}, block {block}");
                }
            }
        }
    }

    #[test]
    fn test_exact_timing_survives_unaligned_chunks_and_reset() {
        for (input_rate, output_rate) in [
            (16000, 48000),
            (48000, 16000),
            (44100, 48000),
            (48000, 44100),
        ] {
            let input: Vec<i16> = (0..2000).map(|i| ((i * 37 % 200) - 100) as i16).collect();
            let mut r = new_resampler(input_rate, output_rate);
            let expected = r.resample(&input);
            assert_eq!(
                expected.len(),
                (input.len() * output_rate).div_ceil(input_rate)
            );
            r.reset();
            let mut got = Vec::new();
            let mut consumed = 0;
            for chunk in input.chunks(7) {
                got.extend(r.resample(chunk));
                consumed += chunk.len();
                assert_eq!(got.len(), (consumed * output_rate).div_ceil(input_rate));
                assert!(r.resample(&[]).is_empty());
            }
            assert_eq!(got, expected, "{input_rate} -> {output_rate}");
            r.reset();
            assert_eq!(r.resample(&input), expected);
        }
    }

    #[cfg(feature = "opus")]
    #[test]
    fn test_resampled_20ms_frames_encode_as_opus() {
        use crate::{Encoder, opus::OpusEncoder};

        for input_rate in [8000, 16000] {
            let mut r = new_resampler(input_rate, 48000);
            let mut encoder = OpusEncoder::new_default();
            for _ in 0..3 {
                let pcm = r.resample(&vec![0; input_rate / 50]);
                assert_eq!(pcm.len(), 960);
                assert!(!encoder.encode(&pcm).is_empty());
            }
        }
    }

    #[test]
    fn test_resample_8k_to_16k() {
        let mut resampler = new_resampler(8000, 16000);
        let input = vec![1000i16; 80];
        let output = resampler.resample(&input);
        assert!(output.len() >= 150 && output.len() <= 170);
        for &s in &output[48..output.len().saturating_sub(48)] {
            assert!((s - 1000).abs() < 100, "Value {} is too far from 1000", s);
        }
    }

    #[test]
    fn test_resample_16k_to_8k() {
        let mut resampler = new_resampler(16000, 8000);
        let input = vec![1000i16; 160];
        let output = resampler.resample(&input);
        assert!(output.len() >= 75 && output.len() <= 85);
        let skip = output.len() / 4;
        for &s in &output[skip..output.len() - skip] {
            assert!((s - 1000).abs() < 100, "Value {} is too far from 1000", s);
        }
    }

    #[test]
    fn test_frequency_response_downsample() {
        let mut resampler = new_resampler(16000, 8000);
        let freq = 2000.0_f32; // Well below 4kHz Nyquist
        let samples: Vec<i16> = (0..160)
            .map(|i| ((i as f32 * freq * 2.0 * PI_F32 / 16000.0).sin() * 10000.0) as i16)
            .collect();

        let output = resampler.resample(&samples);

        // Output should have similar amplitude (allowing for some attenuation)
        let input_rms: f32 = samples
            .iter()
            .map(|&s| (s as f32).powi(2))
            .sum::<f32>()
            .sqrt()
            / samples.len() as f32;
        let output_rms: f32 = output
            .iter()
            .map(|&s| (s as f32).powi(2))
            .sum::<f32>()
            .sqrt()
            / output.len() as f32;

        assert!(
            output_rms > input_rms * 0.7,
            "Too much attenuation: input_rms={}, output_rms={}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn test_aliasing_suppression() {
        let mut resampler = new_resampler(16000, 8000);
        let freq = 7000.0_f32; // Above 4kHz Nyquist of output
        let samples: Vec<i16> = (0..1600)
            .map(|i| ((i as f32 * freq * 2.0 * PI_F32 / 16000.0).sin() * 10000.0) as i16)
            .collect();

        let output = resampler.resample(&samples);

        let output_rms: f32 =
            (output.iter().map(|&s| (s as f32).powi(2)).sum::<f32>() / output.len() as f32).sqrt();
        let input_rms: f32 = 10000.0 / 1.414; // Expected RMS of sine wave with amplitude 10000

        assert!(
            output_rms < input_rms / 50.0,
            "Aliasing not sufficiently suppressed: output_rms={}",
            output_rms
        );
    }

    #[test]
    fn test_performance_48k_to_8k() {
        let mut resampler = new_resampler(48000, 8000);
        let input = vec![0i16; 48000];

        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = resampler.resample(&input);
            resampler.reset();
        }
        let duration = start.elapsed();
        let per_second = duration.as_secs_f64() / iterations as f64;
        println!(
            "Resampling 1s of 48kHz to 8kHz (24 taps) took: {:.4}ms",
            per_second * 1000.0
        );
        assert!(
            per_second < 0.1,
            "Performance regression: {}ms",
            per_second * 1000.0
        );
    }

    #[test]
    fn test_continuity_between_chunks() {
        let input_rate = 16000;
        let output_rate = 8000;

        let freq = 1000.0_f32;
        let total_samples = 3200;
        let input: Vec<i16> = (0..total_samples)
            .map(|i| ((i as f32 * freq * 2.0 * PI_F32 / input_rate as f32).sin() * 5000.0) as i16)
            .collect();

        let mut resampler1 = new_resampler(input_rate, output_rate);
        let output1 = resampler1.resample(&input);

        let mut resampler2 = new_resampler(input_rate, output_rate);
        let mid = input.len() / 2;
        let mut output2 = resampler2.resample(&input[..mid]);
        output2.extend_from_slice(&resampler2.resample(&input[mid..]));

        assert_eq!(output1.len(), output2.len(), "Output lengths differ");

        let max_diff: i16 = output1
            .iter()
            .zip(output2.iter())
            .map(|(a, b)| (a - b).abs())
            .max()
            .unwrap_or(0);

        assert!(
            max_diff < 100,
            "Large discontinuity between chunks: max_diff={}",
            max_diff
        );
    }

    /// The Q15 integer core must match an f32 accumulator using the *same*
    /// quantised coefficients — going integer must not change the result
    /// beyond rounding. (Coefficient quantisation itself is covered by the
    /// passband/aliasing tests above.)
    #[test]
    fn test_q15_core_matches_float_reference() {
        let fin = 16000usize;
        let fout = 48000usize;
        let mut coeffs = vec![0i16; COEFFS_LEN];
        let mut r = Resampler::new(fin, fout, &mut coeffs).unwrap();

        let input: Vec<i16> = (0..2000).map(|i| (((i * 37) % 200 - 100) * 60) as i16).collect();
        let got = r.resample(&input);

        // f32 reference: derive positions from the absolute output index,
        // independently of the streaming accumulator, using the same Q15 taps.
        let taps = TAPS_PER_PHASE;
        let mut hist = [0.0f32; TAPS_PER_PHASE];
        let mut want = Vec::new();
        for (input_index, &s) in input.iter().enumerate() {
            hist.copy_within(1..taps, 0);
            hist[taps - 1] = s as f32;
            while want.len() * fin < (input_index + 1) * fout {
                let phase = (want.len() * fin % fout) * NUM_PHASES / fout;
                let mut acc = 0.0f32;
                for t in 0..taps {
                    acc += hist[t] * (coeffs[phase * taps + t] as f32 / 32768.0);
                }
                want.push(acc.round().clamp(-32768.0, 32767.0) as i16);
            }
        }

        assert_eq!(got.len(), want.len());
        let se: f64 = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum();
        let sig: f64 = want.iter().map(|&b| (b as f64) * (b as f64)).sum();
        let snr = 10.0 * (sig / se.max(1.0)).log10();
        println!("Q15 vs f32-accumulator SNR = {:.1} dB", snr);
        assert!(snr > 60.0, "integer core diverges from float: {:.1} dB", snr);
    }

    /// `BoxedResampler` must produce byte-identical output to the borrowed
    /// `Resampler` fed the same coefficients, and survive being moved +
    /// reused across many calls (stable self-referential borrow).
    #[cfg(feature = "std")]
    #[test]
    fn test_boxed_resampler_matches_borrowed() {
        let input_rate = 48000usize;
        let output_rate = 8000usize;
        let input: Vec<i16> = (0..4800)
            .map(|i| ((i as f32 * 0.05).sin() * 8000.0) as i16)
            .collect();

        let expected = {
            let mut coeffs = vec![0i16; COEFFS_LEN];
            let mut borrowed = Resampler::new(input_rate, output_rate, &mut coeffs).unwrap();
            borrowed.resample(&input)
        };

        let mut boxed = BoxedResampler::new(input_rate, output_rate).unwrap();
        let expected_max = (input.len() * output_rate).div_ceil(input_rate);
        assert_eq!(boxed.max_output_samples(input.len()), expected_max);
        let got = boxed.resample(&input);
        assert_eq!(
            expected, got,
            "BoxedResampler output must match borrowed Resampler"
        );

        // Reuse after move: the coefficient borrow must stay valid across
        // moves. The resampler is a streaming state machine — a second
        // `resample` of the same input keeps the tail of the previous run in
        // its history, so only the output LENGTH is stable across the move.
        let mut moved = boxed;
        let again = moved.resample(&input);
        assert_eq!(expected.len(), again.len());

        // reset() clears the history and phase, so a fresh `resample` of the
        // same input must reproduce the very first output exactly.
        moved.reset();
        let after_reset = moved.resample(&input);
        assert_eq!(expected, after_reset);

        assert!(
            BoxedResampler::new(0, 8000).is_err(),
            "zero input rate must error"
        );
        assert!(
            BoxedResampler::new(48000, 0).is_err(),
            "zero output rate must error"
        );
    }
}
