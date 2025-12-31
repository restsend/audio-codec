use super::{PcmBuf, Sample};
use std::f32::consts::PI;

/// A Polyphase FIR Resampler suitable for VoIP.
pub struct Resampler {
    input_rate: usize,
    output_rate: usize,
    ratio: f64,
    coeffs: Vec<f32>,
    num_phases: usize,
    taps_per_phase: usize,
    history: Vec<f32>,
    current_pos: f64,
}

impl Resampler {
    pub fn new(input_rate: usize, output_rate: usize) -> Self {
        let ratio = output_rate as f64 / input_rate as f64;
        let num_phases = 128;
        let taps_per_phase = 16; // Increased to 16 for SIMD alignment (4x f32)
        let filter_len = num_phases * taps_per_phase;

        let mut raw_coeffs = vec![0.0f32; filter_len];
        let cutoff = if ratio < 1.0 {
            ratio as f32 * 0.45
        } else {
            0.45f32
        };

        let center = (taps_per_phase as f32 - 1.0) / 2.0;

        for p in 0..num_phases {
            let mut sum = 0.0;
            let mut phase_coeffs = vec![0.0f32; taps_per_phase];
            for t in 0..taps_per_phase {
                let x = t as f32 - center - (p as f32 / num_phases as f32);
                let val = if x.abs() < 1e-6 {
                    2.0 * cutoff
                } else {
                    let x_pi = x * PI;
                    (x_pi * 2.0 * cutoff).sin() / x_pi
                };
                let window = 0.54
                    - 0.46
                        * (2.0 * PI * (t as f32 * num_phases as f32 + p as f32)
                            / (filter_len as f32 - 1.0))
                            .cos();
                phase_coeffs[t] = val * window;
                sum += phase_coeffs[t];
            }
            for t in 0..taps_per_phase {
                raw_coeffs[p * taps_per_phase + t] = phase_coeffs[t] / sum;
            }
        }

        Self {
            input_rate,
            output_rate,
            ratio,
            coeffs: raw_coeffs,
            num_phases,
            taps_per_phase,
            history: vec![0.0; taps_per_phase],
            current_pos: 0.0,
        }
    }

    #[inline(always)]
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // Use ARM Neon intrinsics for aarch64 (Apple Silicon)
            unsafe {
                use std::arch::aarch64::*;
                let mut sumv = vdupq_n_f32(0.0);
                for i in (0..16).step_by(4) {
                    let av = vld1q_f32(a.as_ptr().add(i));
                    let bv = vld1q_f32(b.as_ptr().add(i));
                    sumv = vfmaq_f32(sumv, av, bv);
                }
                vaddvq_f32(sumv)
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            // Use AVX for x86_64
            unsafe {
                use std::arch::x86_64::*;
                let av = _mm256_loadu_ps(a.as_ptr());
                let bv = _mm256_loadu_ps(b.as_ptr());
                let mut sumv = _mm256_mul_ps(av, bv);

                let av2 = _mm256_loadu_ps(a.as_ptr().add(8));
                let bv2 = _mm256_loadu_ps(b.as_ptr().add(8));
                sumv = _mm256_add_ps(sumv, _mm256_mul_ps(av2, bv2));

                let x128 = _mm_add_ps(_mm256_extractf128_ps(sumv, 1), _mm256_castps256_ps128(sumv));
                let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
                let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
                _mm_cvtss_f32(x32)
            }
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "avx")
        )))]
        {
            // Fallback to auto-vectorized iterator
            a.iter().zip(b).map(|(x, y)| x * y).sum()
        }
    }

    pub fn resample(&mut self, input: &[Sample]) -> PcmBuf {
        if self.input_rate == self.output_rate {
            return input.to_vec();
        }

        let mut output = Vec::with_capacity((input.len() as f64 * self.ratio) as usize + 1);
        let inv_ratio = 1.0 / self.ratio;
        let taps = self.taps_per_phase;
        let num_phases_f = self.num_phases as f64;

        for &sample in input {
            self.history.copy_within(1..taps, 0);
            self.history[taps - 1] = sample as f32;

            while self.current_pos < 1.0 {
                let phase_idx = (self.current_pos * num_phases_f) as usize;
                let offset = phase_idx * taps;
                let phase_coeffs = &self.coeffs[offset..offset + taps];

                let out_sample = Self::dot_product(phase_coeffs, &self.history);

                output.push(out_sample.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
                self.current_pos += inv_ratio;
            }
            self.current_pos -= 1.0;
        }

        output
    }
}

pub fn resample(input: &[Sample], input_sample_rate: u32, output_sample_rate: u32) -> PcmBuf {
    if input_sample_rate == output_sample_rate {
        return input.to_vec();
    }
    let mut r = Resampler::new(input_sample_rate as usize, output_sample_rate as usize);
    r.resample(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_resample_8k_to_16k() {
        let mut resampler = Resampler::new(8000, 16000);
        let input = vec![1000i16; 80];
        let output = resampler.resample(&input);
        assert!(output.len() >= 150 && output.len() <= 170);
        for &s in &output[20..output.len() - 20] {
            assert!((s - 1000).abs() < 100, "Value {} is too far from 1000", s);
        }
    }

    #[test]
    fn test_resample_16k_to_8k() {
        let mut resampler = Resampler::new(16000, 8000);
        let input = vec![1000i16; 160];
        let output = resampler.resample(&input);
        assert!(output.len() >= 75 && output.len() <= 85);
        for &s in &output[20..output.len() - 20] {
            assert!((s - 1000).abs() < 100, "Value {} is too far from 1000", s);
        }
    }

    #[test]
    fn test_performance() {
        let mut resampler = Resampler::new(48000, 8000);
        let input = vec![0i16; 48000];

        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = resampler.resample(&input);
        }
        let duration = start.elapsed();
        let per_second = duration.as_secs_f64() / iterations as f64;
        println!(
            "Resampling 1s of 48kHz to 8kHz took: {:.4}ms",
            per_second * 1000.0
        );
        assert!(per_second < 0.05);
    }
}
