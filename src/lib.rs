#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate libm;

mod math;
mod matrix;
mod vector;
