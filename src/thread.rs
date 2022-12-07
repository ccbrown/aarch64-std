use aarch64_cpu::registers;
use core::time::Duration;
use tock_registers::interfaces::Readable;

/// Puts the current thread to sleep for at least the specified amount of time.
///
/// The thread may sleep longer than the duration specified due to scheduling specifics or
/// platform-dependent functionality. It will never sleep less.
///
/// This function is blocking, and should not be used in async functions.
pub fn sleep(d: Duration) {
    let freq = registers::CNTFRQ_EL0.get();
    let start = registers::CNTVCT_EL0.get();
    let end = start + (d.as_secs_f64() * freq as f64) as u64;
    // TODO: use a timer or event stream for lower power usage?
    loop {
        if registers::CNTVCT_EL0.get() >= end {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep() {
        sleep(Duration::from_millis(500));
    }
}
