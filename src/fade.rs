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

use crate::Sample;

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

/// Where a decoder is in a loss, to fade its own concealment by and to
/// bring the audio after it back in.
#[derive(Debug, Default)]
pub(crate) struct LossFade {
    /// Samples per channel concealed in the current loss; 0 when not
    /// concealing.
    concealed: usize,
}

impl LossFade {
    /// Fade a concealed frame (interleaved, `channels`) by how far into the
    /// loss it is.
    pub(crate) fn fade(&mut self, out: &mut [Sample], channels: usize, rate: u32) {
        let channels = channels.max(1);
        for (i, frame) in out.chunks_mut(channels).enumerate() {
            let gain = gain(self.concealed + i, rate);
            for s in frame {
                *s = (*s as i32 * gain / 100) as Sample;
            }
        }
        self.concealed += out.len() / channels;
    }

    /// Bring decoded audio ending a loss back in from where the fade had
    /// got to, over `RECOVER_MS`.
    pub(crate) fn end_loss(&mut self, out: &mut [Sample], channels: usize, rate: u32) {
        if self.concealed == 0 {
            return;
        }
        let from = gain(self.concealed, rate);
        self.concealed = 0;
        if from >= 100 {
            return;
        }
        let channels = channels.max(1);
        let len = (rate * RECOVER_MS / 1000) as usize;
        let total = (len + 1) as i32;
        for (i, frame) in out.chunks_mut(channels).take(len).enumerate() {
            let w = (i + 1) as i32;
            let gain = from + (100 - from) * w / total;
            for s in frame {
                *s = (*s as i32 * gain / 100) as Sample;
            }
        }
    }

    /// Forget the loss, as for a decoder reset.
    #[cfg(feature = "opus")]
    pub(crate) fn clear(&mut self) {
        self.concealed = 0;
    }

    /// Samples per channel concealed in the current loss.
    #[cfg(test)]
    pub(crate) fn concealed(&self) -> usize {
        self.concealed
    }
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
