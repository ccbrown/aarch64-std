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
    use core::{arch::asm, convert::Infallible, marker::PhantomData};
    use tock_registers::interfaces::Readable;

    pub struct JoinHandle<T>(PhantomData<T>);

    #[repr(C)]
    #[derive(Default)]
    struct Registers {
        // These are the registers we have to keep track of ourselves. The rest are handled by the
        // compiler based on the inline assembly directives.
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
                    "stp x19, x29, [sp, #-16]!",
                    "stp x1, lr, [sp, #-16]!",
                    "blr {entry}",
                    // if we make it back here, that means the thread's ended
                    "ldp x1, lr, [sp], #16",
                    "ldp x19, x29, [sp], #16",
                    "mov sp, x1",
                    // set did_end = 1
                    "mov x1, #1",
                    "b 2f",

                    "1:",
                    // we're resuming a thread
                    // restore x19, x29, lr, and sp from x8-x11
                    "mov x19, x8",
                    "mov x29, x9",
                    "mov lr, x10",
                    "mov sp, x11",
                    // then jump back to that location
                    "b aarch64_std_unyield",

                    "aarch64_std_yield:",
                    // we're yielding to another thread.
                    // yield_now put our original stack pointer in x0
                    // save x19, x29, lr, and sp to x8-x11
                    "mov x8, x19",
                    "mov x9, x29",
                    "mov x10, lr",
                    "mov x11, sp",
                    // then restore the originals
                    "ldp x19, x29, [x0, #-16]",
                    "ldp x1, lr, [x0, #-32]",
                    "mov sp, x1",
                    // set did_end = 0
                    "mov x1, #0",

                    "2:",
                    entry = in(reg) Self::entry_point,
                    stack = in(reg) stack,
                    inout("x0") args => _,
                    // mark everything possible as clobbered so the compiler can take care of
                    // saving and restoring most of the registers
                    out("x1") did_end, out("x2") _, out("x3") _,
                    out("x4") _, out("x5") _, out("x6") _, out("x7") _,
                    inout("x8") self.registers.x19,
                    inout("x9") self.registers.x29,
                    inout("x10") self.registers.x30,
                    inout("x11") self.registers.sp,
                    out("x12") _, out("x13") _, out("x14") _, out("x15") _,
                    out("x16") _, out("x17") _, out("x18") _,
                    out("x20") _, out("x21") _, out("x22") _, out("x23") _,
                    out("x24") _, out("x25") _, out("x26") _, out("x27") _,
                    out("x28") _,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                    out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                    out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                    out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                    out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                    out("v28") _, out("v29") _, out("v30") _,
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

        fn contribute(&self) {
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
                let state = self.state.lock().unwrap();
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
                    out("x1") _, out("x2") _, out("x3") _,
                    out("x4") _, out("x5") _, out("x6") _, out("x7") _,
                    out("x8") _, out("x9") _, out("x10") _, out("x11") _,
                    out("x12") _, out("x13") _, out("x14") _, out("x15") _,
                    out("x16") _, out("x17") _, out("x18") _,
                    out("x20") _, out("x21") _, out("x22") _, out("x23") _,
                    out("x24") _, out("x25") _, out("x26") _, out("x27") _,
                    out("x28") _,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                    out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                    out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                    out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                    out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                    out("v28") _, out("v29") _, out("v30") _,
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

            let args = RuntimeThreadArgs {
                f: Box::new(|| {
                    // TODO: catch panics and do something with the result?
                    let _ = f();
                }),
            };

            {
                let mut state = self.state.lock().unwrap();
                let id = ThreadId {
                    id: state.next_thread_id,
                    is_external: false,
                };
                state.next_thread_id += 1;
                state.queue.push_back(RuntimeThread {
                    args: Some(args),
                    registers: Default::default(),
                    handle: Thread {
                        inner: Arc::new(ThreadInner {
                            id,
                            name,
                            stack_addr: stack.as_ptr() as u64,
                            stack_size: stack.len() * 16,
                        }),
                    },
                    stack,
                });
            }

            JoinHandle(PhantomData)
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// let other_thread = thread::spawn(|| {
    ///     thread::current().id()
    /// });
    ///
    /// let other_thread_id = other_thread.join().unwrap();
    /// assert!(thread::current().id() != other_thread_id);
    /// ```
    ///
    /// [`id`]: Thread::id
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
            Self {
                name: None,
                stack_size: 2 * 1024,
            }
        }

        /// Names the thread-to-be. Currently the name is used for identification
        /// only in panic messages.
        ///
        /// The name must not contain null bytes (`\0`).
        ///
        /// For more information about named threads, see
        /// [this module-level documentation][naming-threads].
        ///
        /// # Examples
        ///
        /// ```
        /// use std::thread;
        ///
        /// let builder = thread::Builder::new()
        ///     .name("foo".into());
        ///
        /// let handler = builder.spawn(|| {
        ///     assert_eq!(thread::current().name(), Some("foo"))
        /// }).unwrap();
        ///
        /// handler.join().unwrap();
        /// ```
        pub fn name(mut self, name: String) -> Builder {
            self.name = Some(name);
            self
        }

        /// Sets the size of the stack (in bytes) for the new thread.
        ///
        /// # Examples
        ///
        /// ```
        /// use std::thread;
        ///
        /// let builder = thread::Builder::new().stack_size(32 * 1024);
        /// ```
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
        pub fn spawn<F, T>(self, f: F) -> Result<JoinHandle<T>, Infallible>
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

    /// This function should be called by a hardware or OS thread. The native thread will
    /// contribute its CPU time to the runtime's green threads and returns if there are no
    /// green threads that currently need to be driven (at which point you may just want to
    /// call this function again).
    pub fn contribute() {
        GLOBAL_RUNTIME.contribute();
    }
}

#[cfg(feature = "alloc")]
pub use runtime::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep() {
        sleep(Duration::from_millis(500));
    }

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

    #[test]
    fn test_spawn() {
        Builder::new()
            .name("foo".into())
            .spawn(|| {
                let t = current();
                assert_eq!(t.name().unwrap(), "foo");

                println!("i'm the thread!");
                yield_now();
                println!("i'm still the thread!");
            })
            .unwrap();

        Builder::new()
            .name("bar".into())
            .spawn(|| {
                let t = current();
                assert_eq!(t.name().unwrap(), "bar");

                println!("i'm the other thread!");
            })
            .unwrap();

        contribute();
    }
}
