//! High-performance algorithms for L-VM core operations.

#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::vec::Vec;
pub mod dsu;

pub use dsu::{fast_connected_components, DisjointSet};
