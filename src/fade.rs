//! How concealment fades through a long loss, after ITU-T G.711 Appendix I:
//! the first 10 ms play at full level, then the level drops by 20% per
//! 10 ms, reaching silence 60 ms in. Concealment only guesses at the audio,
//! and the longer the loss the worse the guess, so a long one goes quiet
//! rather than drone on. The first audio decoded after a loss is brought
//! back in over `RECOVER_MS`.

/// Concealment plays at full level for this long, then fades.
const FULL_LEVEL_MS: u32 = 10;
/// Gain lost per 10 ms after `FULL_LEVEL_MS`, in percent; silence at 60 ms.
const FADE_PERCENT_PER_10MS: u32 = 20;
/// Blend into the first decoded frame after a loss.
pub(crate) const RECOVER_MS: u32 = 4;

/// Gain in percent, `t` samples (per channel, at `rate`) into a loss.
pub(crate) fn gain(t: usize, rate: u32) -> i32 {
    let full = (rate * FULL_LEVEL_MS / 1000) as usize;
    if t < full {
        return 100;
    }
    let per_10ms = ((rate / 100) as usize).max(1);
    let faded = (t - full) * FADE_PERCENT_PER_10MS as usize / per_10ms;
    100 - faded.min(100) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_for_10_ms_then_silent_at_60_ms() {
        for rate in [8000, 16000, 48000] {
            let ms = |ms: u32| (rate * ms / 1000) as usize;
            assert_eq!(gain(0, rate), 100);
            assert_eq!(gain(ms(10) - 1, rate), 100);
            assert_eq!(gain(ms(20), rate), 80);
            assert_eq!(gain(ms(35), rate), 50);
            assert_eq!(gain(ms(60), rate), 0);
            assert_eq!(gain(ms(500), rate), 0);
        }
    }
}
