#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Useful synchronization primitives.
pub mod sync;

/// Thread utilities, including cooperative green threads.
pub mod thread;

/// Temporal quantification.
pub mod time;
