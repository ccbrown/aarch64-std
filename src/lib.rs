#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod sync;
pub mod thread;
pub mod time;
