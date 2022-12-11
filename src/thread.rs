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
        #[cfg(feature = "alloc")]
        yield_now();
        if registers::CNTVCT_EL0.get() >= end {
            return;
        }
    }
}

#[cfg(feature = "alloc")]
mod runtime {
    use crate::sync::Mutex;
    use alloc::{boxed::Box, collections::LinkedList, string::String, sync::Arc, vec::Vec};
    use core::{any::Any, arch::asm, convert::Infallible, time::Duration};
    use tock_registers::interfaces::Readable;

    pub type Result<T> = core::result::Result<T, Box<dyn Any + Send + 'static>>;

    pub struct JoinHandle<T> {
        result: Arc<Mutex<Option<Result<T>>>>,
        thread: Thread,
    }

    impl<T> JoinHandle<T> {
        /// Extracts a handle to the underlying thread.
        #[must_use]
        pub fn thread(&self) -> &Thread {
            &self.thread
        }

        /// Waits for the associated thread to finish.
        ///
        /// This function will return immediately if the associated thread has already finished.
        ///
        /// In terms of [atomic memory orderings],  the completion of the associated
        /// thread synchronizes with this function returning. In other words, all
        /// operations performed by that thread [happen
        /// before](https://doc.rust-lang.org/nomicon/atomics.html#data-accesses) all
        /// operations that happen after `join` returns.
        ///
        /// If the associated thread panics, [`Err`] is returned with the parameter given
        /// to [`panic!`].
        ///
        /// [`Err`]: crate::result::Result::Err
        /// [atomic memory orderings]: crate::sync::atomic
        ///
        /// # Panics
        ///
        /// This function may panic on some platforms if a thread attempts to join
        /// itself or otherwise may create a deadlock with joining threads.
        pub fn join(self) -> Result<T> {
            loop {
                if let Some(result) = self.result.lock().unwrap().take() {
                    return result;
                }
                yield_now();
            }
        }

        /// Checks if the associated thread has finished running its main function.
        ///
        /// `is_finished` supports implementing a non-blocking join operation, by checking
        /// `is_finished`, and calling `join` if it returns `true`. This function does not block. To
        /// block while waiting on the thread to finish, use [`join`][Self::join].
        ///
        /// This might return `true` for a brief moment after the thread's main
        /// function has returned, but before the thread itself has stopped running.
        /// However, once this returns `true`, [`join`][Self::join] can be expected
        /// to return quickly, without blocking for any significant amount of time.
        pub fn is_finished(&self) -> bool {
            Arc::strong_count(&self.result) == 1
        }
    }

    #[repr(C)]
    #[derive(Default)]
    struct Registers {
        // These are the registers we have to keep track of ourselves. The rest are handled by the
        // compiler based on the inline assembly directives.
        x18: u64,
        x19: u64,
        x29: u64,
        x30: u64,
        sp: u64,
    }

    #[repr(C, align(16))]
    #[derive(Default)]
    struct SixteenBytes([u8; 16]);

    struct RuntimeThread {
        args: Option<RuntimeThreadArgs>,
        stack: Vec<SixteenBytes>,
        registers: Registers,
        handle: Thread,
    }

    enum RunStatus {
        Yielded,
        Ended,
    }

    struct RuntimeThreadArgs {
        f: Box<dyn FnOnce() + Send>,
    }

    impl RuntimeThread {
        fn entry_point(args: *mut RuntimeThreadArgs) {
            let args = unsafe { Box::from_raw(args) };
            (args.f)();
        }

        #[allow(named_asm_labels)]
        #[inline(never)]
        fn run(&mut self) -> RunStatus {
            unsafe {
                let stack = self.stack.as_mut_ptr().offset(self.stack.len() as _);
                let args = self
                    .args
                    .take()
                    .map(|args| Box::into_raw(Box::new(args)))
                    .unwrap_or_else(core::ptr::null_mut);
                let did_end: u64;
                asm!(
                    // see if we're starting a new thread or resuming one
                    "cmp x0, #0",
                    "beq 1f",

                    // we're starting a new thread
                    "mov x1, sp",
                    "mov sp, {stack}",
                    "str x18, [sp, #-16]!",
                    "stp x19, x29, [sp, #-16]!",
                    "stp x1, lr, [sp, #-16]!",
                    "blr {entry}",
                    // if we make it back here, that means the thread's ended
                    "ldp x1, lr, [sp], #16",
                    "ldp x19, x29, [sp], #16",
                    "ldr x18, [sp], #16",
                    "mov sp, x1",
                    // set did_end = 1
                    "mov x1, #1",
                    "b 2f",

                    "1:",
                    // we're resuming a thread
                    // restore x18, x19, x29, lr, and sp from x8-x12
                    "mov x18, x8",
                    "mov x19, x9",
                    "mov x29, x10",
                    "mov lr, x11",
                    "mov sp, x12",
                    // then jump back to that location
                    "b aarch64_std_unyield",

                    "aarch64_std_yield:",
                    // we're yielding to another thread.
                    // yield_now put our original stack pointer in x0
                    // save x18, x19, x29, lr, and sp to x8-x12
                    "mov x8, x18",
                    "mov x9, x19",
                    "mov x10, x29",
                    "mov x11, lr",
                    "mov x12, sp",
                    // then restore the originals
                    "ldr x18, [x0, #-16]",
                    "ldp x19, x29, [x0, #-32]",
                    "ldp x1, lr, [x0, #-48]",
                    "mov sp, x1",
                    // set did_end = 0
                    "mov x1, #0",

                    "2:",
                    entry = in(reg) Self::entry_point,
                    stack = in(reg) stack,
                    inout("x0") args => _,
                    out("x1") did_end,
                    inout("x8") self.registers.x18,
                    inout("x9") self.registers.x19,
                    inout("x10") self.registers.x29,
                    inout("x11") self.registers.x30,
                    inout("x12") self.registers.sp,
                    // mark everything possible as clobbered so the compiler can take care of
                    // saving and restoring most of the registers
                    out("x20") _, out("x21") _, out("x22") _, out("x23") _,
                    out("x24") _, out("x25") _, out("x26") _, out("x27") _,
                    out("x28") _,
                    clobber_abi("C")
                );
                if did_end == 1 {
                    RunStatus::Ended
                } else {
                    RunStatus::Yielded
                }
            }
        }
    }

    struct Runtime {
        state: Mutex<RuntimeState>,
    }

    struct RuntimeState {
        next_thread_id: u64,
        queue: LinkedList<RuntimeThread>,
        active_threads: Vec<Thread>,
    }

    impl Runtime {
        const fn new() -> Self {
            Self {
                state: Mutex::new(RuntimeState {
                    next_thread_id: 1,
                    queue: LinkedList::new(),
                    active_threads: Vec::new(),
                }),
            }
        }

        unsafe fn contribute(&self) {
            loop {
                match {
                    let mut state = self.state.lock().unwrap();
                    let thread = state.queue.pop_front();
                    if let Some(thread) = &thread {
                        state.active_threads.push(thread.handle.clone());
                    }
                    thread
                } {
                    Some(mut thread) => match thread.run() {
                        RunStatus::Yielded => {
                            let mut state = self.state.lock().unwrap();
                            state.queue.push_back(thread);
                        }
                        RunStatus::Ended => {
                            let mut state = self.state.lock().unwrap();
                            let idx = state
                                .active_threads
                                .iter()
                                .position(|t| t.id() == thread.handle.id())
                                .unwrap();
                            state.active_threads.swap_remove(idx);
                        }
                    },
                    None => return,
                }
            }
        }

        fn current(&self) -> Thread {
            let stack: u64;
            unsafe {
                asm!(
                    "mov {stack}, sp",
                    stack = out(reg) stack,
                );
            }
            let state = self.state.lock().unwrap();
            for t in &state.active_threads {
                if stack >= t.inner.stack_addr
                    && stack < t.inner.stack_addr + t.inner.stack_size as u64
                {
                    return t.clone();
                }
            }
            Thread {
                inner: Arc::new(ThreadInner {
                    id: ThreadId {
                        id: aarch64_cpu::registers::TPIDRRO_EL0.get(),
                        is_external: true,
                    },
                    name: None,
                    stack_addr: 0,
                    stack_size: 0,
                }),
            }
        }

        #[allow(named_asm_labels)]
        #[inline(never)]
        fn yield_now(&self) {
            let t = {
                let state = self.state.lock_impl(false).unwrap();
                if state.queue.is_empty() {
                    // There are no pending threads.
                    unsafe { asm!("yield") };
                    return;
                }

                let stack: u64;
                unsafe {
                    asm!(
                        "mov {stack}, sp",
                        stack = out(reg) stack,
                    );
                }

                match state.active_threads.iter().find(|t| {
                    stack >= t.inner.stack_addr
                        && stack < t.inner.stack_addr + t.inner.stack_size as u64
                }) {
                    Some(t) => t.clone(),
                    None => {
                        // This isn't one of our threads.
                        unsafe { asm!("yield") };
                        return;
                    }
                }
            };

            unsafe {
                asm!(
                    "b aarch64_std_yield",
                    "aarch64_std_unyield:",
                    inout("x0") t.inner.stack_addr + t.inner.stack_size as u64 => _,
                    // mark everything possible as clobbered so the compiler can take care of
                    // saving and restoring most of the registers
                    out("x20") _, out("x21") _, out("x22") _, out("x23") _,
                    out("x24") _, out("x25") _, out("x26") _, out("x27") _,
                    out("x28") _,
                    clobber_abi("C")
                );
            }
        }

        /// Spawns a thread.
        fn spawn<F, T>(&self, f: F, name: Option<String>, stack_size: usize) -> JoinHandle<T>
        where
            F: FnOnce() -> T,
            F: Send + 'static,
            T: Send + 'static,
        {
            let stack_size_div_16 = (stack_size + 15) / 16;
            let mut stack = Vec::with_capacity(stack_size_div_16);
            stack.resize_with(stack_size_div_16, Default::default);

            let result = Arc::new(Mutex::new(None));

            let args = RuntimeThreadArgs {
                f: {
                    let result = result.clone();
                    Box::new(move || {
                        // TODO: is there a way we can catch panics?
                        let ret = f();
                        *result.lock().unwrap() = Some(Ok(ret));
                    })
                },
            };

            let mut state = self.state.lock().unwrap();

            let id = ThreadId {
                id: state.next_thread_id,
                is_external: false,
            };
            state.next_thread_id += 1;

            let handle = Thread {
                inner: Arc::new(ThreadInner {
                    id,
                    name,
                    stack_addr: stack.as_ptr() as u64,
                    stack_size: stack.len() * 16,
                }),
            };

            state.queue.push_back(RuntimeThread {
                args: Some(args),
                registers: Default::default(),
                handle: handle.clone(),
                stack,
            });

            JoinHandle {
                thread: handle,
                result,
            }
        }
    }

    static GLOBAL_RUNTIME: Runtime = Runtime::new();

    /// A unique identifier for a running thread.
    ///
    /// A `ThreadId` is an opaque object that uniquely identifies each thread
    /// created during the lifetime of a process. `ThreadId`s are guaranteed not to
    /// be reused, even when a thread terminates. `ThreadId`s are under the control
    /// of Rust's standard library and there may not be any relationship between
    /// `ThreadId` and the underlying platform's notion of a thread identifier --
    /// the two concepts cannot, therefore, be used interchangeably. A `ThreadId`
    /// can be retrieved from the [`id`] method on a [`Thread`].
    #[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
    pub struct ThreadId {
        id: u64,
        is_external: bool,
    }

    /// A handle to a thread.
    ///
    /// Threads are represented via the `Thread` type, which you can get in one of
    /// two ways:
    ///
    /// * By spawning a new thread, e.g., using the [`thread::spawn`][`spawn`]
    ///   function, and calling [`thread`][`JoinHandle::thread`] on the
    ///   [`JoinHandle`].
    /// * By requesting the current thread, using the [`thread::current`] function.
    ///
    /// The [`thread::current`] function is available even for threads not spawned
    /// by the APIs of this module.
    ///
    /// There is usually no need to create a `Thread` struct yourself, one
    /// should instead use a function like `spawn` to create new threads, see the
    /// docs of [`Builder`] and [`spawn`] for more details.
    ///
    /// [`thread::current`]: current
    #[derive(Clone, Debug)]
    pub struct Thread {
        inner: Arc<ThreadInner>,
    }

    #[derive(Clone, Debug)]
    pub struct ThreadInner {
        id: ThreadId,
        name: Option<String>,
        stack_addr: u64,
        stack_size: usize,
    }

    impl Thread {
        /// Gets the thread’s unique identifier.
        pub fn id(&self) -> ThreadId {
            self.inner.id
        }

        /// Gets the thread’s unique identifier.
        pub fn name(&self) -> Option<&str> {
            self.inner.name.as_ref().map(|s| s.as_str())
        }

        /// Atomically makes the handle's token available if it is not already.
        ///
        /// Every thread is equipped with some basic low-level blocking support, via
        /// the [`park`][park] function and the `unpark()` method. These can be
        /// used as a more CPU-efficient implementation of a spinlock.
        ///
        /// See the [park documentation][park] for more details.
        #[inline]
        pub fn unpark(&self) {
            // TODO: a more efficient implementation?
        }
    }

    /// Thread factory, which can be used in order to configure the properties of
    /// a new thread.
    ///
    /// Methods can be chained on it in order to configure it.
    ///
    /// The two configurations available are:
    ///
    /// - [`name`]: specifies an [associated name for the thread][naming-threads]
    /// - [`stack_size`]: specifies the [desired stack size for the thread][stack-size]
    ///
    /// The [`spawn`] method will take ownership of the builder and create an
    /// [`io::Result`] to the thread handle with the given configuration.
    ///
    /// The [`thread::spawn`] free function uses a `Builder` with default
    /// configuration and [`unwrap`]s its return value.
    ///
    /// You may want to use [`spawn`] instead of [`thread::spawn`], when you want
    /// to recover from a failure to launch a thread, indeed the free function will
    /// panic where the `Builder` method will return a [`io::Result`].
    #[must_use = "must eventually spawn the thread"]
    #[derive(Debug)]
    pub struct Builder {
        name: Option<String>,
        stack_size: usize,
    }

    impl Builder {
        /// Generates the base configuration for spawning a thread, from which configuration methods can be chained.
        pub fn new() -> Self {
            const DEFAULT_STACK_SIZE: usize = 8 * 1024;
            Self {
                name: None,
                stack_size: DEFAULT_STACK_SIZE,
            }
        }

        /// Names the thread-to-be. Currently the name is used for identification
        /// only in panic messages.
        ///
        /// The name must not contain null bytes (`\0`).
        ///
        /// For more information about named threads, see
        /// [this module-level documentation][naming-threads].
        pub fn name(mut self, name: String) -> Builder {
            self.name = Some(name);
            self
        }

        /// Sets the size of the stack (in bytes) for the new thread.
        pub fn stack_size(mut self, size: usize) -> Builder {
            self.stack_size = size;
            self
        }

        /// Spawns a new thread by taking ownership of the `Builder`, and returns an
        /// [`io::Result`] to its [`JoinHandle`].
        ///
        /// The spawned thread may outlive the caller (unless the caller thread
        /// is the main thread; the whole process is terminated when the main
        /// thread finishes). The join handle can be used to block on
        /// termination of the spawned thread, including recovering its panics.
        ///
        /// For a more complete documentation see [`thread::spawn`][`spawn`].
        pub fn spawn<F, T>(self, f: F) -> core::result::Result<JoinHandle<T>, Infallible>
        where
            F: FnOnce() -> T,
            F: Send + 'static,
            T: Send + 'static,
        {
            Ok(GLOBAL_RUNTIME.spawn(f, self.name, self.stack_size))
        }
    }

    /// Spawns a new thread, returning a [`JoinHandle`] for it.
    ///
    /// The join handle provides a [`join`] method that can be used to join the spawned
    /// thread. If the spawned thread panics, [`join`] will return an [`Err`] containing
    /// the argument given to [`panic!`].
    ///
    /// If the join handle is dropped, the spawned thread will implicitly be *detached*.
    /// In this case, the spawned thread may no longer be joined.
    /// (It is the responsibility of the program to either eventually join threads it
    /// creates or detach them; otherwise, a resource leak will result.)
    ///
    /// This call will create a thread using default parameters of [`Builder`], if you
    /// want to specify the stack size or the name of the thread, use this API
    /// instead.
    ///
    /// As you can see in the signature of `spawn` there are two constraints on
    /// both the closure given to `spawn` and its return value, let's explain them:
    ///
    /// - The `'static` constraint means that the closure and its return value
    ///   must have a lifetime of the whole program execution. The reason for this
    ///   is that threads can outlive the lifetime they have been created in.
    ///
    ///   Indeed if the thread, and by extension its return value, can outlive their
    ///   caller, we need to make sure that they will be valid afterwards, and since
    ///   we *can't* know when it will return we need to have them valid as long as
    ///   possible, that is until the end of the program, hence the `'static`
    ///   lifetime.
    /// - The [`Send`] constraint is because the closure will need to be passed
    ///   *by value* from the thread where it is spawned to the new thread. Its
    ///   return value will need to be passed from the new thread to the thread
    ///   where it is `join`ed.
    ///   As a reminder, the [`Send`] marker trait expresses that it is safe to be
    ///   passed from thread to thread. [`Sync`] expresses that it is safe to have a
    ///   reference be passed from thread to thread.
    pub fn spawn<F, T>(f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        Builder::new().spawn(f).expect("failed to spawn thread")
    }

    /// Gets a handle to the thread that invokes it.
    pub fn current() -> Thread {
        GLOBAL_RUNTIME.current()
    }

    /// Cooperatively gives up a timeslice to the scheduler.
    ///
    /// For multithreading to work effectively, threads must call this function whenever they are
    /// willing to be swapped out.
    ///
    /// If called within the context of a spawned thread, another pending thread will be swapped
    /// in. Otherwise, this will evaluate to an assembly YIELD instruction.
    ///
    /// Many functions within this crate such as [`sleep`] have built-in calls to this function.
    pub fn yield_now() {
        GLOBAL_RUNTIME.yield_now();
    }

    /// This is a non-standard function that should be called by a hardware or OS thread in order to drive spawned threads.
    ///
    /// The native thread will contribute its CPU time to the runtime's green threads and returns
    /// if there are no green threads that currently need to be driven (at which point you may just
    /// want to call this function again).
    ///
    /// # Safety
    /// User space threads can't reliably detect stack overflows. Some systems have protections in
    /// place that will crash the program on overflow, but others will simply have undefined
    /// behavior. To use spawned threads safely, you must ensure that your stack sizes are big
    /// enough to never overflow.
    pub unsafe fn contribute() {
        GLOBAL_RUNTIME.contribute();
    }

    /// Blocks unless or until the current thread's token is made available.
    ///
    /// A call to `park` does not guarantee that the thread will remain parked
    /// forever, and callers should be prepared for this possibility.
    ///
    /// # park and unpark
    ///
    /// Every thread is equipped with some basic low-level blocking support, via the
    /// [`thread::park`][`park`] function and [`thread::Thread::unpark`][`unpark`]
    /// method. [`park`] blocks the current thread, which can then be resumed from
    /// another thread by calling the [`unpark`] method on the blocked thread's
    /// handle.
    ///
    /// Conceptually, each [`Thread`] handle has an associated token, which is
    /// initially not present:
    ///
    /// * The [`thread::park`][`park`] function blocks the current thread unless or
    ///   until the token is available for its thread handle, at which point it
    ///   atomically consumes the token. It may also return *spuriously*, without
    ///   consuming the token. [`thread::park_timeout`] does the same, but allows
    ///   specifying a maximum time to block the thread for.
    ///
    /// * The [`unpark`] method on a [`Thread`] atomically makes the token available
    ///   if it wasn't already. Because the token is initially absent, [`unpark`]
    ///   followed by [`park`] will result in the second call returning immediately.
    ///
    /// In other words, each [`Thread`] acts a bit like a spinlock that can be
    /// locked and unlocked using `park` and `unpark`.
    ///
    /// Notice that being unblocked does not imply any synchronization with someone
    /// that unparked this thread, it could also be spurious.
    /// For example, it would be a valid, but inefficient, implementation to make both [`park`] and
    /// [`unpark`] return immediately without doing anything.
    ///
    /// The API is typically used by acquiring a handle to the current thread,
    /// placing that handle in a shared data structure so that other threads can
    /// find it, and then `park`ing in a loop. When some desired condition is met, another
    /// thread calls [`unpark`] on the handle.
    ///
    /// The motivation for this design is twofold:
    ///
    /// * It avoids the need to allocate mutexes and condvars when building new
    ///   synchronization primitives; the threads already provide basic
    ///   blocking/signaling.
    ///
    /// * It can be implemented very efficiently on many platforms.
    ///
    /// [`unpark`]: Thread::unpark
    /// [`thread::park_timeout`]: park_timeout
    pub fn park() {
        // TODO: a more efficient implementation?
        yield_now();
    }

    /// Blocks unless or until the current thread's token is made available or
    /// the specified duration has been reached (may wake spuriously).
    ///
    /// The semantics of this function are equivalent to [`park`][park] except
    /// that the thread will be blocked for roughly no longer than `dur`. This
    /// method should not be used for precise timing due to anomalies such as
    /// preemption or platform differences that might not cause the maximum
    /// amount of time waited to be precisely `dur` long.
    ///
    /// See the [park documentation][park] for more details.
    pub fn park_timeout(_dur: Duration) {
        // TODO: a more efficient implementation?
        yield_now();
    }
}

#[cfg(feature = "alloc")]
pub use runtime::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "alloc")]
    use crate::sync::Mutex;
    #[cfg(feature = "alloc")]
    use alloc::sync::Arc;

    #[test]
    fn test_sleep() {
        sleep(Duration::from_millis(500));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_current_with_os_threads() {
        let a = std::thread::spawn(|| std::thread::current())
            .join()
            .unwrap();
        let b = std::thread::spawn(|| std::thread::current())
            .join()
            .unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_spawn() {
        let v = Arc::new(Mutex::new(Vec::new()));

        let foo = Builder::new()
            .name("foo".into())
            .spawn({
                let v = v.clone();
                move || {
                    let t = current();
                    assert_eq!(t.name().unwrap(), "foo");

                    v.lock().unwrap().push(1);
                    yield_now();
                    loop {
                        let mut v = v.lock().unwrap();
                        if v.len() == 2 {
                            v.push(3);
                            break;
                        }
                    }

                    "foo"
                }
            })
            .unwrap();

        let bar = Builder::new()
            .name("bar".into())
            .spawn({
                let v = v.clone();
                move || {
                    let t = current();
                    assert_eq!(t.name().unwrap(), "bar");

                    loop {
                        let mut v = v.lock().unwrap();
                        if v.len() == 1 {
                            v.push(2);
                            break;
                        }
                    }

                    "bar"
                }
            })
            .unwrap();

        unsafe { contribute() };

        assert!(foo.is_finished());
        assert_eq!(foo.join().unwrap(), "foo");

        assert!(bar.is_finished());
        assert_eq!(bar.join().unwrap(), "bar");

        assert_eq!(*v.lock().unwrap(), vec![1, 2, 3]);
    }
}
