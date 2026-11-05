//! Packet loss concealment by pitch waveform repetition, after ITU-T G.711
//! Appendix I, for codecs with none of their own (G.711, G.722).
//!
//! A lost frame is filled by repeating the last pitch period of the audio
//! before it, continuing the waveform where it stopped. The period's last
//! quarter is overlap-added with the quarter before it, so each repetition
//! runs into the next without a step. (Appendix I also smooths the seam
//! with the last audio played, by holding output back 3.75 ms; this plays
//! without delay, so that seam is only as smooth as the pitch estimate.)
//!
//! Long losses fade out as Appendix I has it (see [`crate::fade`]): full
//! level for 10 ms, silence at 60 ms. The first frame decoded after a loss
//! is crossfaded with the continued repetition.
//!
//! Heap-free: the history is a fixed array sized for 16 kHz.

use super::Sample;
use crate::fade::{self, RECOVER_MS};

/// Pitch range searched, in Hz (as in G.711 Appendix I).
const MIN_PITCH_HZ: u32 = 66;
const MAX_PITCH_HZ: u32 = 200;
/// Window correlated when estimating the pitch.
const CORR_MS: u32 = 20;

/// Highest sample rate the fixed buffers are sized for.
const MAX_RATE: u32 = 16000;
const MAX_PERIOD: usize = (MAX_RATE / MIN_PITCH_HZ) as usize + 1;
const HISTORY: usize = (MAX_RATE * CORR_MS / 1000) as usize + MAX_PERIOD;
/// Full-level audio kept from just before each extrapolated frame.
pub(crate) const LEAD: usize = 24;

pub(crate) struct PitchConcealer {
    rate: u32,
    /// The most recent audio played, oldest first.
    history: [Sample; HISTORY],
    /// Valid samples at the end of `history`.
    filled: usize,
    /// The pitch period repeated during a loss.
    period: [Sample; MAX_PERIOD],
    period_len: usize,
    /// Read position in `period`.
    pos: usize,
    /// Samples concealed in the current loss; 0 when not concealing.
    concealed: usize,
    /// The full-level audio leading into the latest extrapolated frame:
    /// the audio played before the loss, then the unfaded repetition.
    lead: [Sample; LEAD],
    /// The last `LEAD` samples of the latest unfaded extrapolation.
    tail: [Sample; LEAD],
}

impl PitchConcealer {
    /// `rate` must be at most 16 kHz.
    pub(crate) fn new(rate: u32) -> Self {
        debug_assert!(rate <= MAX_RATE);
        Self {
            rate,
            history: [0; HISTORY],
            filled: 0,
            period: [0; MAX_PERIOD],
            period_len: 0,
            pos: 0,
            concealed: 0,
            lead: [0; LEAD],
            tail: [0; LEAD],
        }
    }

    fn ms(&self, ms: u32) -> usize {
        (self.rate * ms / 1000) as usize
    }

    /// History needed to pick a pitch period.
    fn needed(&self) -> usize {
        self.ms(CORR_MS) + self.ms(1000 / MIN_PITCH_HZ)
    }

    fn remember(&mut self, samples: &[Sample]) {
        let n = samples.len();
        if n >= HISTORY {
            self.history.copy_from_slice(&samples[n - HISTORY..]);
        } else {
            self.history.copy_within(n.., 0);
            self.history[HISTORY - n..].copy_from_slice(samples);
        }
        self.filled = (self.filled + n).min(HISTORY);
    }

    /// The last `n` samples remembered, oldest first.
    #[cfg(test)]
    pub(crate) fn recent(&self, n: usize) -> &[Sample] {
        &self.history[HISTORY - n.min(HISTORY)..]
    }

    /// Record decoded audio. A frame ending a loss is first crossfaded, in
    /// place, with the continued repetition.
    pub(crate) fn on_decoded(&mut self, samples: &mut [Sample]) {
        if samples.is_empty() {
            // nothing played: not the end of a loss
            return;
        }
        if self.concealed > 0 && self.period_len > 0 {
            let len = self.ms(RECOVER_MS).min(samples.len());
            let gain = self.gain(self.concealed);
            let total = (len + 1) as i32;
            for (i, s) in samples[..len].iter_mut().enumerate() {
                let t = self.next_period_sample() as i32 * gain / 100;
                let w = (i + 1) as i32;
                *s = ((*s as i32 * w + t * (total - w)) / total) as Sample;
            }
        }
        self.concealed = 0;
        self.remember(samples);
    }

    /// Fill `out` with the repeated pitch period at full level, continuing
    /// the waveform. Returns `false`, leaving `out` silent, when there's
    /// not enough audio yet to conceal from.
    pub(crate) fn extrapolate(&mut self, out: &mut [Sample]) -> bool {
        if self.filled < self.needed() {
            out.fill(0);
            return false;
        }
        // also when a loss began before there was audio to pick from: the
        // silence played meanwhile went into the history
        if self.concealed == 0 || self.period_len == 0 {
            self.pick_period();
            self.lead.copy_from_slice(&self.history[HISTORY - LEAD..]);
        } else {
            self.lead = self.tail;
        }
        for s in out.iter_mut() {
            *s = self.next_period_sample();
        }
        let n = out.len().min(LEAD);
        self.tail.copy_within(n.., 0);
        self.tail[LEAD - n..].copy_from_slice(&out[out.len() - n..]);
        true
    }

    /// The full-level audio leading into the latest extrapolated frame,
    /// oldest first.
    pub(crate) fn lead(&self) -> &[Sample; LEAD] {
        &self.lead
    }

    /// Whether the next concealed frame will be heard at all, or the loss
    /// has faded out.
    pub(crate) fn audible(&self) -> bool {
        self.gain(self.concealed) > 0
    }

    /// Apply the loss's fade to an extrapolated frame and record it as
    /// played.
    pub(crate) fn fade(&mut self, out: &mut [Sample]) {
        for (i, s) in out.iter_mut().enumerate() {
            *s = (*s as i32 * self.gain(self.concealed + i) / 100) as Sample;
        }
        self.concealed += out.len();
        self.remember(out);
    }

    /// Conceal one lost frame: [`Self::extrapolate`] then [`Self::fade`].
    pub(crate) fn conceal(&mut self, out: &mut [Sample]) {
        self.extrapolate(out);
        self.fade(out);
    }

    /// Gain in percent, `t` samples into a loss.
    fn gain(&self, t: usize) -> i32 {
        fade::gain(t, self.rate)
    }

    /// The repetition `k` samples past the last one extrapolated, without
    /// advancing it.
    pub(crate) fn peek(&self, k: usize) -> Sample {
        if self.period_len == 0 {
            return 0;
        }
        self.period[(self.pos + k) % self.period_len]
    }

    fn next_period_sample(&mut self) -> Sample {
        let s = self.period[self.pos];
        self.pos = (self.pos + 1) % self.period_len;
        s
    }

    /// Pick the pitch period to repeat: the lag whose preceding window
    /// correlates best with the latest one.
    fn pick_period(&mut self) {
        let min_lag = self.ms(1000 / MAX_PITCH_HZ).max(1);
        let max_lag = self.ms(1000 / MIN_PITCH_HZ);
        let win = self.ms(CORR_MS);
        let h = &self.history;
        let target = &h[HISTORY - win..];
        let mut best = (min_lag, f32::MIN);
        for lag in min_lag..=max_lag {
            let cand = &h[HISTORY - win - lag..HISTORY - lag];
            let (mut xy, mut yy) = (0f32, 0f32);
            for (x, y) in target.iter().zip(cand) {
                xy += *x as f32 * *y as f32;
                yy += *y as f32 * *y as f32;
            }
            // normalized correlation, compared squared to avoid a sqrt
            let score = if yy > 0.0 { xy * xy.abs() / yy } else { 0.0 };
            if score > best.1 {
                best = (lag, score);
            }
        }
        let p = best.0;
        self.period_len = p;
        self.period[..p].copy_from_slice(&h[HISTORY - p..]);
        // the period wraps from its end, the last sample played, back to
        // its start: fade its last quarter into the quarter before its
        // start, the audio that led into the start
        let ola = p / 4;
        let before = &h[HISTORY - p - ola..HISTORY - p];
        let total = (ola + 1) as i32;
        let tail = &mut self.period[p - ola..p];
        for (i, (t, b)) in tail.iter_mut().zip(before).enumerate() {
            let w = (i + 1) as i32;
            *t = ((*t as i32 * (total - w) + *b as i32 * w) / total) as Sample;
        }
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16000;
    const FRAME: usize = 320;

    /// A 150 Hz tone, continuous across frames.
    fn tone(frame: usize, out: &mut [Sample]) {
        for (i, s) in out.iter_mut().enumerate() {
            let t = (frame * FRAME + i) as f32 / RATE as f32;
            *s = (8000.0 * libm_sin(2.0 * core::f32::consts::PI * 150.0 * t)) as Sample;
        }
    }

    // no_std test builds have no f32::sin
    fn libm_sin(x: f32) -> f32 {
        let mut x = x % (2.0 * core::f32::consts::PI);
        if x > core::f32::consts::PI {
            x -= 2.0 * core::f32::consts::PI;
        }
        // Taylor series, accurate enough on [-pi, pi] for a test tone
        let x2 = x * x;
        x * (1.0
            - x2 / 6.0
                * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0 * (1.0 - x2 / 110.0)))))
    }

    fn rms(s: &[Sample]) -> f32 {
        let sum: f32 = s.iter().map(|x| *x as f32 * *x as f32).sum();
        let mean = sum / s.len() as f32;
        // Newton's method, no sqrt in no_std
        let mut r = mean.max(1.0);
        for _ in 0..30 {
            r = 0.5 * (r + mean / r);
        }
        r
    }

    fn max_step(prev: Sample, s: &[Sample]) -> i32 {
        let mut last = prev as i32;
        let mut max = 0;
        for x in s {
            max = max.max((*x as i32 - last).abs());
            last = *x as i32;
        }
        max
    }

    fn primed() -> PitchConcealer {
        let mut plc = PitchConcealer::new(RATE);
        let mut buf = [0; FRAME];
        for f in 0..5 {
            tone(f, &mut buf);
            plc.on_decoded(&mut buf);
        }
        plc
    }

    #[test]
    fn silence_before_there_is_audio_to_conceal_from() {
        let mut plc = PitchConcealer::new(RATE);
        let mut out = [1; FRAME];
        assert!(!plc.extrapolate(&mut out));
        assert!(out.iter().all(|s| *s == 0));
    }

    #[test]
    fn concealment_continues_the_waveform() {
        let mut plc = primed();
        let last = *plc.recent(1).last().unwrap();
        let mut out = [0; FRAME];
        plc.conceal(&mut out);
        let mut reference = [0; FRAME];
        tone(5, &mut reference);
        assert!(rms(&out) > 0.5 * rms(&reference), "rms {}", rms(&out));
        // a 150 Hz tone at 8000 moves at most ~470 per sample
        assert!(max_step(last, &out) < 1000, "step {}", max_step(last, &out));
    }

    #[test]
    fn concealment_continues_from_any_phase() {
        // the loss may cut the waveform anywhere in its cycle: the
        // concealment must carry on from that point, across its own
        // repetitions too
        for offset in (0..FRAME).step_by(13) {
            let mut plc = PitchConcealer::new(RATE);
            let mut buf = [0; FRAME];
            for f in 0..5 {
                tone(f, &mut buf);
                // shift the tone so frames end at a different phase
                let shifted: [Sample; FRAME] = core::array::from_fn(|i| {
                    let mut one = [0; FRAME];
                    tone(f + (i + offset) / FRAME, &mut one);
                    one[(i + offset) % FRAME]
                });
                buf = shifted;
                plc.on_decoded(&mut buf);
            }
            let last = *plc.recent(1).last().unwrap();
            let mut out = [0; FRAME];
            plc.extrapolate(&mut out);
            let step = max_step(last, &out);
            assert!(step < 1000, "offset {offset}: step {step}");
        }
    }

    #[test]
    fn long_loss_fades_to_silence() {
        let mut plc = primed();
        let mut frames = [[0; FRAME]; 4];
        for f in frames.iter_mut() {
            plc.conceal(f);
        }
        assert!(rms(&frames[1]) < rms(&frames[0]));
        assert!(frames[3].iter().all(|s| *s == 0));
    }

    #[test]
    fn recovery_is_crossfaded() {
        let mut plc = primed();
        let mut concealed = [0; FRAME];
        plc.conceal(&mut concealed);
        // the real audio resumes out of phase with the concealment
        let mut real = [0; FRAME];
        tone(6, &mut real);
        for s in real.iter_mut() {
            *s = -*s;
        }
        plc.on_decoded(&mut real);
        let last = *concealed.last().unwrap();
        assert!(
            max_step(last, &real[..64]) < 2000,
            "step {}",
            max_step(last, &real[..64])
        );
    }

    #[test]
    fn works_at_8khz() {
        let mut plc = PitchConcealer::new(8000);
        let mut buf = [0; 160];
        for f in 0..5 {
            for (i, s) in buf.iter_mut().enumerate() {
                let t = (f * 160 + i) as f32 / 8000.0;
                *s = (8000.0 * libm_sin(2.0 * core::f32::consts::PI * 150.0 * t)) as Sample;
            }
            plc.on_decoded(&mut buf);
        }
        let mut out = [0; 160];
        assert!(plc.extrapolate(&mut out));
        assert!(rms(&out) > 2000.0);
    }
}
