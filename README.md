# aarch64-std

aarch64-std implements components from the Rust standard library in a way suitable for `no_std` or bare metal ARM applications.

## Design Goals

In order:

1. Run on any aarch64 platform at EL0. Anything from microcontrollers to Zynq UltraScale+ MPSoCs to M1 Macs should just work.
2. Mimic the standard library APIs as closely as possible. For the most part these modules are drop-in replacements for the standard library.
3. Perform as efficiently as possible.

## Highlights

- `sync`
  - `Mutex`
- `thread`
  - `sleep`
  - `spawn`
    - Uses cooperative green threads.
    - As many cores as you'd like can participate using `thread::contribute`.
    - Threads yield via the standard `thread::yield_now` function.
- `time`
  - `Instant`

## Cargo Features

- "alloc" enables functionality which requires the use of the [alloc](https://doc.rust-lang.org/alloc/index.html) crate. It's enabled by default, but may be disabled if you don't have an allocator available.
