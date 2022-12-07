use aarch64_cpu::registers;
use tock_registers::interfaces::Readable;

pub use core::time::*;

/// A measurement of a monotonically nondecreasing clock. Opaque and useful only with Duration.
///
/// Instants are always guaranteed, barring platform bugs, to be no less than any previously
/// measured instant when created, and are often useful for tasks such as measuring benchmarks or
/// timing how long an operation takes.
///
/// Note, however, that instants are not guaranteed to be steady. In other words, each tick of the
/// underlying clock might not be the same length (e.g. some seconds may be longer than others). An
/// instant may jump forwards or experience time dilation (slow down or speed up), but it will
/// never go backwards.
///
/// Instants are opaque types that can only be compared to one another. There is no method to get
/// “the number of seconds” from an instant. Instead, it only allows measuring the duration between
/// two instants (or comparing two instants).
#[derive(Clone, Copy, Debug, Hash)]
pub struct Instant {
    ticks: u64,
}

impl Instant {
    /// Returns an instant corresponding to “now”.
    pub fn now() -> Self {
        Self {
            ticks: registers::CNTVCT_EL0.get(),
        }
    }

    /// Returns the amount of time elapsed from another instant to this one, or zero duration if
    /// that instant is later than this one.
    pub fn duration_since(&self, earlier: Self) -> Duration {
        let freq = registers::CNTFRQ_EL0.get();
        let ticks = self.ticks - earlier.ticks;
        Duration::from_secs_f64(ticks as f64 / freq as f64)
    }

    /// Returns the amount of time elapsed since this instant was created.
    pub fn elapsed(&self) -> Duration {
        Self::now().duration_since(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thread::sleep;

    #[test]
    fn test_instant() {
        let start = Instant::now();
        const SLEEP_DURATION: Duration = Duration::from_millis(500);
        sleep(SLEEP_DURATION);
        assert!(start.elapsed() >= SLEEP_DURATION);
    }
}
