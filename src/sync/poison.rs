use core::convert::Infallible;

/// An enumeration of possible errors associated with a [`TryLockResult`] which
/// can occur while trying to acquire a lock, from the [`try_lock`] method on a
/// [`Mutex`] or the [`try_read`] and [`try_write`] methods on an [`RwLock`].
///
/// [`try_lock`]: crate::sync::Mutex::try_lock
/// [`try_read`]: crate::sync::RwLock::try_read
/// [`try_write`]: crate::sync::RwLock::try_write
/// [`Mutex`]: crate::sync::Mutex
/// [`RwLock`]: crate::sync::RwLock
pub enum TryLockError {
    /// The lock could not be acquired at this time because the operation would
    /// otherwise block.
    WouldBlock,
}

#[cfg(feature = "alloc")]
impl alloc::fmt::Debug for TryLockError {
    fn fmt(&self, f: &mut alloc::fmt::Formatter<'_>) -> alloc::fmt::Result {
        match *self {
            TryLockError::WouldBlock => "WouldBlock".fmt(f),
        }
    }
}

#[cfg(feature = "alloc")]
impl alloc::fmt::Display for TryLockError {
    fn fmt(&self, f: &mut alloc::fmt::Formatter<'_>) -> alloc::fmt::Result {
        match *self {
            TryLockError::WouldBlock => "try_lock failed because the operation would block",
        }
        .fmt(f)
    }
}

pub type LockResult<Guard> = Result<Guard, Infallible>;

/// A type alias for the result of a nonblocking locking method.
///
/// For more information, see [`LockResult`]. A `TryLockResult` doesn't
/// necessarily hold the associated guard in the [`Err`] type as the lock might not
/// have been acquired for other reasons.
pub type TryLockResult<Guard> = Result<Guard, TryLockError>;
