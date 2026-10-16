use core::f64::consts::PI as PI_F64;

use fearless_simd::{Level, Simd, dispatch};

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
// The phase index is kept in 8 bits (see `resample_simd`).
const _: () = assert!(NUM_PHASES == 256);
/// The 24-tap window is accumulated as two `i32` parts: the 8 centre taps
/// (`MID_TAPS`, where nearly all of the filter's weight sits) and the 16
/// outer taps. Each part is a whole number of 128-bit multiply-adds.
const MID_TAPS: core::ops::Range<usize> = 8..16;
/// Largest `sum(|coefficient|)` within either part for which an `i32`
/// accumulator provably cannot overflow: `|acc| <= 32768 * sum(|coeff|) < 2^31`.
const MAX_PART_ABS_COEFF_SUM: i32 = 65535;

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
/// history, integer accumulator, integer phase accumulator). This matters on
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
    /// The last `TAPS_PER_PHASE` input samples, oldest first.
    history: [i16; TAPS_PER_PHASE],
    /// Ticks from the newest input sample to the next output sample. It can
    /// exceed one input sample when downsampling.
    ///
    /// Timing runs on an integer grid of `input_rate * output_rate` ticks per
    /// second: one input sample is `output_rate` ticks and one output sample
    /// is `input_rate` ticks, so the timing is exact and never rounded.
    pos: usize,
    /// SIMD level the filter loop is compiled for.
    level: Level,
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

            let mut part_abs_sums = [0i32; 2];
            for t in 0..taps_per_phase {
                // Quantise the normalised tap to Q15.
                let q = fround((phase_coeffs[t] / sum) * 32768.0);
                let q = q.clamp(-32768.0, 32767.0) as i16;
                coeffs[p * taps_per_phase + t] = q;
                part_abs_sums[MID_TAPS.contains(&t) as usize] += (q as i32).abs();
            }
            // The filter loop accumulates each part in `i32`; refuse the
            // (degenerate) filters for which that could overflow.
            if part_abs_sums.iter().any(|&s| s > MAX_PART_ABS_COEFF_SUM) {
                return Err(CodecError::InvalidInput);
            }
        }

        Ok(Self {
            input_rate,
            output_rate,
            coeffs,
            history: [0; TAPS_PER_PHASE],
            pos: 0,
            level: default_level(),
        })
    }

    pub fn input_rate(&self) -> usize {
        self.input_rate
    }

    pub fn output_rate(&self) -> usize {
        self.output_rate
    }

    /// Q15 dot product of the 24-tap window against one phase.
    ///
    /// The two parts of the window ([`MID_TAPS`] and the rest) are each
    /// accumulated in `i32` (each `i16`×`i16` product fits, and
    /// [`MAX_PART_ABS_COEFF_SUM`], enforced in [`Self::new`], bounds the part
    /// sums) and combined in `i64`. The full 24-tap sum can exceed `i32` for
    /// full-scale input (`sum(|coeff|)` is ~2.1), so this is the cheapest way
    /// to stay *exactly* equal to a plain `i64` accumulation. Plain widening
    /// multiply-add loops like these are what the compiler turns into
    /// `pmaddwd`/`smlal` when compiled for a SIMD target feature level.
    ///
    /// Takes fixed-size array refs so the outer loop's slice↔array `try_into`
    /// is the only length check.
    #[inline(always)]
    fn dot_q15(history: &[i16; TAPS_PER_PHASE], coeffs: &[i16; TAPS_PER_PHASE]) -> i32 {
        let mut mid = 0i32;
        for i in MID_TAPS {
            mid = mid.wrapping_add((history[i] as i32) * (coeffs[i] as i32));
        }
        let mut outer = 0i32;
        for i in 0..MID_TAPS.start {
            outer = outer.wrapping_add((history[i] as i32) * (coeffs[i] as i32));
        }
        for i in MID_TAPS.end..TAPS_PER_PHASE {
            outer = outer.wrapping_add((history[i] as i32) * (coeffs[i] as i32));
        }
        ((mid as i64 + outer as i64) >> Q15_SHIFT) as i32
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

        // Exact number of outputs this call must produce: outputs sit
        // `input_rate` ticks apart starting at `pos`, and the call consumes
        // `input.len() * output_rate` ticks. Anything else means output sizes
        // drift across chunk boundaries (the 0.4.8 bug: 20 ms frames came out
        // with 959/961 samples). Pinned by `debug_assert_eq!` below.
        let expected: u128 = {
            let ticks = (input.len() as u128) * (self.output_rate as u128);
            if ticks > self.pos as u128 {
                (ticks - self.pos as u128).div_ceil(self.input_rate as u128)
            } else {
                0
            }
        };

        // Dispatch once per call so the whole loop is compiled with the
        // selected target features and `dot_q15` inlines into it.
        let level = self.level;
        let written = dispatch!(level, simd => self.resample_simd(simd, input, out))?;

        // The tick-grid timing is exact by construction; this assert pins it
        // so a future change to the stepping can never silently reintroduce
        // wrong frame sizes (debug builds and tests only; free in release).
        debug_assert_eq!(
            written as u128,
            expected,
            "resampler produced {written} samples, exact timing requires {expected} \
             for {} inputs ({input_rate} -> {output_rate} at pos {}): output \
             sizes would drift across chunk boundaries",
            input.len(),
            self.pos,
            input_rate = self.input_rate,
            output_rate = self.output_rate,
        );

        Ok(written)
    }

    #[inline(always)]
    fn resample_simd<S: Simd>(
        &mut self,
        _simd: S,
        input: &[Sample],
        out: &mut [Sample],
    ) -> Result<usize, CodecError> {
        /// Input samples processed per block. Staging a whole block before
        /// filtering (instead of pushing one sample at a time into the
        /// window) avoids store-forwarding stalls: the wide window loads
        /// would otherwise hit a just-written narrow store on every output.
        const BLOCK: usize = 256;

        // Tick periods (see `pos`): `step` per output, `interval` per input.
        let (step, interval) = (self.input_rate, self.output_rate);
        // One output step in whole input samples plus a fraction of one, and
        // that fraction in phase units: `step_frac * 256 = dq * interval + dr`.
        // `new` checked that `interval * NUM_PHASES` fits.
        let (step_whole, step_frac) = (step / interval, step % interval);
        let (dq, dr) = (
            step_frac * NUM_PHASES / interval,
            step_frac * NUM_PHASES % interval,
        );

        let coeffs: &[[i16; TAPS_PER_PHASE]; NUM_PHASES] =
            self.coeffs.as_chunks().0[..NUM_PHASES].try_into().unwrap();
        // `[history | block]`: the window for an output at block index `i`
        // is `buf[i + 1..i + 1 + TAPS_PER_PHASE]`, i.e. the last
        // `TAPS_PER_PHASE` samples up to and including sample `i`.
        let mut buf = [0i16; TAPS_PER_PHASE + BLOCK];
        let mut written = 0usize;

        for chunk in input.chunks(BLOCK) {
            let n = chunk.len();
            buf[..TAPS_PER_PHASE].copy_from_slice(&self.history);
            buf[TAPS_PER_PHASE..TAPS_PER_PHASE + n].copy_from_slice(chunk);

            // `pos` counts ticks from the block's first sample; an output is
            // due every `step` ticks before the block end.
            let pos = self.pos as u64;
            let end = n as u64 * interval as u64;
            let count = if pos < end {
                (end - pos).div_ceil(step as u64) as usize
            } else {
                0
            };
            let out_block = out
                .get_mut(written..written + count)
                .ok_or(CodecError::BufferTooSmall)?;

            // The output position, kept as the input sample it follows plus
            // the filter phase and the remainder below it:
            // `pos = idx * interval + frac`, `frac * 256 = phase * interval + rem`.
            // Main picks the phase as `frac * 256 / interval`; tracking it with
            // carries gives the same phase without a division per output.
            let mut idx = self.pos / interval;
            let frac = self.pos % interval;
            let mut phase = frac * NUM_PHASES / interval;
            let mut rem = frac * NUM_PHASES % interval;

            for slot in out_block {
                // `idx < n <= BLOCK` and `phase < NUM_PHASES`, so the masks never
                // change them; they let the compiler drop the bounds checks.
                let i = idx & (BLOCK - 1);
                let window = buf[i + 1..i + 1 + TAPS_PER_PHASE].try_into().unwrap();
                let out_sample = Self::dot_q15(window, &coeffs[phase & (NUM_PHASES - 1)]);
                *slot = out_sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

                // Advance one output (`step` ticks).
                rem += dr;
                if rem >= interval {
                    rem -= interval;
                    phase += 1;
                }
                phase += dq;
                idx += step_whole + phase / NUM_PHASES;
                phase %= NUM_PHASES;
            }

            written += count;
            self.pos = (pos + count as u64 * step as u64 - end) as usize;
            self.history.copy_from_slice(&buf[n..n + TAPS_PER_PHASE]);
        }

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
                debug_assert!(
                    n <= max,
                    "resampler emitted {n} samples but `max_output_samples` \
                     said {max}; buffer sizing invariant broken"
                );
                out.truncate(n);
                out
            }
            Err(e) => {
                // Unreachable while the buffer is sized by
                // `max_output_samples` (an exact upper bound). Panic in debug
                // builds and tests so a sizing/count regression can never
                // silently turn into empty (e.g. Opus) frames; release keeps
                // the documented empty-vec-on-error behaviour.
                if cfg!(debug_assertions) {
                    panic!("resample_into failed inside `resample`: {e:?}");
                }
                Vec::new()
            }
        }
    }

    pub fn reset(&mut self) {
        self.history.fill(0);
        self.pos = 0;
    }
}

/// Best SIMD level available: runtime-detected when possible (`std`, wasm32),
/// otherwise the level implied by the compile-time target features.
///
/// Only AVX-512 is downgraded, to AVX2: the filter is at most 128-bit
/// `pmaddwd` wide either way, but at the AVX-512 level LLVM also
/// autovectorizes the block staging with 512-bit instructions. On Ice Lake
/// (Xeon Gold 6354) that made every ratio 20-25% slower than AVX2. Other
/// levels are used as detected.
fn default_level() -> Level {
    let level = Level::try_detect().unwrap_or_else(Level::baseline);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if level.as_avx512().is_some()
        && let Some(avx2) = level.as_avx2()
    {
        return avx2.level();
    }
    level
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
                    assert_eq!(
                        n,
                        output.len(),
                        "{input_rate} -> {output_rate}, block {block}"
                    );
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
            // Sweep chunk sizes around the history length (24) so every
            // phase-accumulator alignment is exercised: the running output
            // count must always be exactly ceil(consumed * out / in).
            for chunk_size in [1, 2, 3, 5, 7, 11, 23, 24, 25, 47, 88, 160, 319] {
                r.reset();
                let mut got = Vec::new();
                let mut consumed = 0;
                for chunk in input.chunks(chunk_size) {
                    got.extend(r.resample(chunk));
                    consumed += chunk.len();
                    assert_eq!(
                        got.len(),
                        (consumed * output_rate).div_ceil(input_rate),
                        "{input_rate} -> {output_rate}, chunk {chunk_size}, consumed {consumed}"
                    );
                    assert!(r.resample(&[]).is_empty());
                }
                assert_eq!(
                    got, expected,
                    "{input_rate} -> {output_rate}, chunk {chunk_size}"
                );
            }
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

        let input: Vec<i16> = (0..2000)
            .map(|i| (((i * 37) % 200 - 100) * 60) as i16)
            .collect();
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
        assert!(
            snr > 60.0,
            "integer core diverges from float: {:.1} dB",
            snr
        );
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

    const RATIOS: [(usize, usize); 8] = [
        (8000, 16000),
        (16000, 8000),
        (48000, 8000),
        (8000, 48000),
        (44100, 48000),
        (48000, 44100),
        (8000, 22050),
        (22050, 16000),
    ];

    /// Deterministic test signals: sine + LCG noise, and a full-scale square
    /// wave whose filter overshoot exercises output saturation.
    fn test_signals(len: usize) -> [Vec<i16>; 2] {
        let mut seed = 0x1234_5678u32;
        let noisy: Vec<i16> = (0..len)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = ((seed >> 16) as i16 as f32) * 0.2;
                ((i as f32 * 0.07).sin() * 15000.0 + noise) as i16
            })
            .collect();
        let square: Vec<i16> = (0..len)
            .map(|i| {
                if (i / 10) % 2 == 0 {
                    i16::MAX
                } else {
                    i16::MIN
                }
            })
            .collect();
        [noisy, square]
    }

    /// Straightforward scalar implementation: per-sample shifting window,
    /// `i64` accumulator, and each output's position derived from its
    /// absolute index rather than a streaming accumulator. The optimised loop
    /// must match it exactly: there is no float rounding to excuse
    /// differences.
    fn reference_resample(coeffs: &[i16], fin: usize, fout: usize, input: &[i16]) -> Vec<i16> {
        let mut history = [0i16; TAPS_PER_PHASE];
        let mut out = Vec::new();
        for (input_index, &sample) in input.iter().enumerate() {
            history.copy_within(1.., 0);
            history[TAPS_PER_PHASE - 1] = sample;
            while out.len() * fin < (input_index + 1) * fout {
                let phase = (out.len() * fin % fout) * NUM_PHASES / fout;
                let c = &coeffs[phase * TAPS_PER_PHASE..(phase + 1) * TAPS_PER_PHASE];
                let acc: i64 = c
                    .iter()
                    .zip(&history)
                    .map(|(&c, &h)| c as i64 * h as i64)
                    .sum();
                out.push((acc >> Q15_SHIFT).clamp(i16::MIN as i64, i16::MAX as i64) as i16);
            }
        }
        out
    }

    /// Every SIMD level reachable on this machine must match the scalar
    /// reference bit for bit.
    #[test]
    fn test_simd_matches_scalar_reference() {
        let levels = [Level::baseline(), default_level(), Level::new()];
        for (from, to) in RATIOS {
            for input in test_signals(2000) {
                for level in levels {
                    let mut r = new_resampler(from, to);
                    r.level = level;
                    let expected = reference_resample(r.coeffs, from, to, &input);
                    let got = r.resample(&input);
                    assert_eq!(expected, got, "{from}->{to} {level:?}");
                }
            }
        }
    }

    /// Each `i32` part-accumulator is only exact while the part's absolute
    /// coefficient sum stays within [`MAX_PART_ABS_COEFF_SUM`]; `new`
    /// enforces it. Check that realistic (and extreme) ratios are accepted
    /// and that the worst part leaves headroom.
    #[test]
    fn test_part_coeff_sums_fit_i32_accumulator() {
        let mut worst = 0i32;
        let rates = [
            8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000, 192000,
        ];
        // The filter depends only on the ratio (cutoff scales with it when
        // downsampling), so also sweep it finely, including extremes.
        let sweep = (1..=100).flat_map(|k| [(48000, 480 * k), (480 * k, 48000)]);
        let pairs = rates
            .iter()
            .flat_map(|&from| rates.iter().map(move |&to| (from, to)))
            .chain(sweep);
        for (from, to) in pairs {
            let mut coeffs = vec![0i16; COEFFS_LEN];
            Resampler::new(from, to, &mut coeffs).unwrap_or_else(|e| panic!("{from}->{to}: {e:?}"));
            for phase in coeffs.chunks(TAPS_PER_PHASE) {
                let abs = |c: &i16| (*c as i32).abs();
                let mid: i32 = phase[MID_TAPS].iter().map(abs).sum();
                let outer: i32 = phase.iter().map(abs).sum::<i32>() - mid;
                worst = worst.max(mid).max(outer);
            }
        }
        println!("worst per-part sum(|coeff|) = {worst} (limit {MAX_PART_ABS_COEFF_SUM})");
        assert!(worst <= MAX_PART_ABS_COEFF_SUM);
    }

    /// The full 24-tap sum can exceed `i32` (sum(|coeff|) is ~2.1 for
    /// upsampling filters). Sign-matched full-scale input drives every phase
    /// to that worst case; the result must be the exact `i64` value, not a
    /// wrapped one.
    #[test]
    fn test_dot_q15_exact_beyond_i32_range() {
        let r = new_resampler(8000, 48000);
        let mut beyond_i32 = 0;
        for phase in r.coeffs.chunks(TAPS_PER_PHASE) {
            let phase: &[i16; TAPS_PER_PHASE] = phase.try_into().unwrap();
            let window: [i16; TAPS_PER_PHASE] =
                core::array::from_fn(|i| if phase[i] < 0 { i16::MIN } else { i16::MAX });
            let exact: i64 = window
                .iter()
                .zip(phase)
                .map(|(&h, &c)| h as i64 * c as i64)
                .sum();
            beyond_i32 += (exact > i32::MAX as i64) as usize;
            assert_eq!(
                Resampler::dot_q15(&window, phase) as i64,
                exact >> Q15_SHIFT,
                "phase {phase:?}"
            );
        }
        assert!(
            beyond_i32 > 0,
            "test no longer reaches the i32 overflow range"
        );
    }

    /// Streaming in arbitrary chunk sizes (smaller than, around and larger
    /// than the history length and the internal block size) must be
    /// bit-identical to a single call.
    #[test]
    fn test_chunked_is_bit_identical() {
        let [input, _] = test_signals(3000);
        for (from, to) in RATIOS {
            let whole = new_resampler(from, to).resample(&input);
            for chunk in [1, 2, 7, 23, 24, 25, 47, 160, 255, 256, 257, 1000] {
                let mut r = new_resampler(from, to);
                let mut chunked = Vec::new();
                for part in input.chunks(chunk) {
                    chunked.extend_from_slice(&r.resample(part));
                }
                assert_eq!(whole, chunked, "{from}->{to} chunk={chunk}");
            }
        }
    }

    /// Constant input must come out at the same level for every ratio
    /// (each polyphase branch has ~unity DC gain, up to Q15 quantisation).
    #[test]
    fn test_dc_gain_all_ratios() {
        for (from, to) in RATIOS {
            for level in [1000i16, -20000, i16::MAX] {
                let output = new_resampler(from, to).resample(&vec![level; 2000]);
                // Skip the filter warm-up (history starts zeroed).
                let warmup = (TAPS_PER_PHASE * to).div_ceil(from) + 1;
                // 24 taps quantised to 1/32768 each: error well under 0.1%.
                let tolerance = 2 + (level as i32).abs() / 1000;
                for (i, &s) in output.iter().enumerate().skip(warmup) {
                    assert!(
                        (s as i32 - level as i32).abs() <= tolerance,
                        "{from}->{to}: sample {i} = {s}, expected ~{level}"
                    );
                }
            }
        }
    }

    /// Output never exceeds `max_output_samples`, for any chunk length and
    /// any carried-over position.
    #[test]
    fn test_max_output_samples_is_upper_bound() {
        for (from, to) in RATIOS {
            let mut r = new_resampler(from, to);
            let input = vec![100i16; 200];
            for n in 1..=200 {
                let mut out = vec![0i16; r.max_output_samples(n)];
                r.resample_into(&input[..n], &mut out)
                    .unwrap_or_else(|e| panic!("{from}->{to} n={n}: {e:?}"));
            }
        }
    }

    #[test]
    fn test_output_buffer_too_small() {
        let mut r = new_resampler(8000, 16000);
        let mut out = [0i16; 10];
        assert!(matches!(
            r.resample_into(&[0i16; 160], &mut out),
            Err(CodecError::BufferTooSmall)
        ));

        // Same-rate passthrough checks the size too.
        let mut r = new_resampler(8000, 8000);
        assert!(matches!(
            r.resample_into(&[0i16; 160], &mut out),
            Err(CodecError::BufferTooSmall)
        ));
    }

    #[test]
    fn test_same_rate_passthrough() {
        let [input, _] = test_signals(500);
        let mut r = new_resampler(16000, 16000);
        assert_eq!(r.max_output_samples(input.len()), input.len());
        assert_eq!(r.resample(&input), input);
    }

    #[test]
    fn test_new_rejects_bad_args() {
        let mut small = vec![0i16; COEFFS_LEN - 1];
        assert!(matches!(
            Resampler::new(8000, 16000, &mut small),
            Err(CodecError::BufferTooSmall)
        ));
        let mut coeffs = vec![0i16; COEFFS_LEN];
        assert!(matches!(
            Resampler::new(0, 16000, &mut coeffs),
            Err(CodecError::InvalidInput)
        ));
        assert!(matches!(
            Resampler::new(8000, 0, &mut coeffs),
            Err(CodecError::InvalidInput)
        ));
    }

    /// `reset` must restore the exact initial state, even mid-stream.
    #[test]
    fn test_reset_mid_stream() {
        let [input, _] = test_signals(1000);
        let mut r = new_resampler(44100, 48000);
        let fresh = r.resample(&input);
        // 13 samples leaves partial history and a non-zero `pos`.
        r.resample(&input[..13]);
        r.reset();
        assert_eq!(r.resample(&input), fresh);
    }
}
