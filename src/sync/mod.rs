pub use core::sync::*;

#[cfg(feature = "alloc")]
pub use alloc::sync::*;

mod mutex;
pub use mutex::*;

mod poison;
pub use poison::*;
