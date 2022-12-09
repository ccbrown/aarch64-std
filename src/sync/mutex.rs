use super::poison::*;
use core::{
    arch::asm,
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

#[derive(Debug, Default)]
pub struct Mutex<T: ?Sized> {
    is_locked: u32,
    inner: UnsafeCell<T>,
}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    pub const fn new(t: T) -> Self {
        Mutex {
            is_locked: 0,
            inner: UnsafeCell::new(t),
        }
    }
}

unsafe impl<T: ?Sized + Send> Send for Mutex<T> {}
unsafe impl<T: ?Sized + Send> Sync for Mutex<T> {}

impl<T: ?Sized> Mutex<T> {
    /// Acquires a mutex, blocking the current thread until it is able to do so.
    ///
    /// This function will block the local thread until it is available to acquire
    /// the mutex. Upon returning, the thread is the only thread with the lock
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    ///
    /// The exact behavior on locking a mutex in the thread which already holds
    /// the lock is left unspecified. However, this function will not return on
    /// the second call (it might panic or deadlock, for example).
    ///
    /// # Errors
    ///
    /// Currently this function cannot fail. The standard library's Mutex may fail if there is a
    /// panic while the lock is held, but without the standard library we currently have no good
    /// way to detect panics. Poisoning may be added at a later time.
    ///
    /// # Panics
    ///
    /// This function might panic when called if the lock is already held by
    /// the current thread.
    pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
        self.lock_impl(true)
    }

    pub(crate) fn lock_impl(&self, _yield_on_fail: bool) -> LockResult<MutexGuard<'_, T>> {
        aarch64_cpu::asm::sevl();
        loop {
            aarch64_cpu::asm::wfe();
            match self.try_lock() {
                Ok(g) => return Ok(g),
                Err(TryLockError::WouldBlock) => {
                    #[cfg(feature = "alloc")]
                    if _yield_on_fail {
                        crate::thread::yield_now();
                    }
                    continue;
                }
            }
        }
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then [`Err`] is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    ///
    /// # Errors
    ///
    /// If the mutex could not be acquired because it is already locked, then
    /// this call will return the [`WouldBlock`] error.
    pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
        let mut result: u32;
        unsafe {
            asm!(
                "ldaxr {result:w}, [{is_locked_addr}]",
                "cmp {result:w}, 0",
                "bne 1f",
                "mov {tmp:w}, 1",
                "stlxr {result:w}, {tmp:w}, [{is_locked_addr}]",
                "1:",
                is_locked_addr = in(reg) &self.is_locked as *const u32 as u64,
                tmp = out(reg) _,
                result = out(reg) result,
                options(nostack),
            );
        }
        if result == 0 {
            Ok(MutexGuard {
                lock: self,
                _make_unsend: PhantomData,
            })
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Consumes this mutex, returning the underlying data.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error instead.
    pub fn into_inner(self) -> LockResult<T>
    where
        T: Sized,
    {
        Ok(self.inner.into_inner())
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the `Mutex` mutably, no actual locking needs to
    /// take place -- the mutable borrow statically guarantees no locks exist.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error instead.
    pub fn get_mut(&mut self) -> LockResult<&mut T> {
        Ok(self.inner.get_mut())
    }
}

#[derive(Debug)]
pub struct MutexGuard<'a, T: ?Sized + 'a> {
    lock: &'a Mutex<T>,
    _make_unsend: PhantomData<*const u8>,
}

unsafe impl<T: ?Sized + Sync> Sync for MutexGuard<'_, T> {}

impl<T: ?Sized> Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.inner.get() }
    }
}

impl<T: ?Sized> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.inner.get() }
    }
}

impl<'a, T: ?Sized> Drop for MutexGuard<'a, T> {
    fn drop(&mut self) {
        // TODO: poison the lock if there's a way to find out if we're panicking
        unsafe {
            asm!(
                "stlr {1:w}, [{0}]",
                "sev",
                in(reg) &self.lock.is_locked as *const u32 as *mut u32,
                in(reg) 0u32,
                options(nostack),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_contention() {
        let n = Mutex::new(1);

        {
            let mut guard = n.lock().unwrap();
            assert!(n.try_lock().is_err());
            *guard += 1;
            assert_eq!(*guard, 2);
        }

        {
            let mut guard = n.lock().unwrap();
            assert!(n.try_lock().is_err());
            *guard += 1;
            assert_eq!(*guard, 3);
        }

        {
            let mut guard = n.try_lock().unwrap();
            *guard += 1;
            assert_eq!(*guard, 4);
        }
    }
}
