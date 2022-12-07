# aarch64-std

aarch64-std implements components from the Rust standard library in a way suitable for `no_std` or bare metal ARM applications.

It's designed to mimic the standard library APIs as closely as possible and work out of the box on any aarch64 platform, from microcontrollers to Zynq UltraScale+ MPSoCs to M1 Macs.

## Cargo Features

- "alloc" enables functionality which requires the use of the [alloc](https://doc.rust-lang.org/alloc/index.html) crate. It's enabled by default, but may be disabled if you don't have an allocator available.
