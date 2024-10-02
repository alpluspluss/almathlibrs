#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate libm;

pub const PI: f32 = std::f32::consts::PI;
pub const TAU: f32 = 6.28318530718;
pub const E: f32 = std::f32::consts::E;
pub const SQRT_2: f32 = std::f32::consts::SQRT_2;
pub const LN_2: f32 = std::f32::consts::LN_2;
pub const DEG_TO_RAD: f32 = PI / 180.0;
pub const RAD_TO_DEG: f32 = 180.0 / PI;
pub const EPSILON: f32 = 1e-5;

pub fn min(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}
pub fn max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

//noinspection ALL
pub fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + t * (end - start)
}
pub fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}
pub fn approx_zero(a: f32, epsilon: f32) -> bool {
    a.abs() < epsilon
}
pub fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * DEG_TO_RAD
}
pub fn radians_to_degrees(radians: f32) -> f32 {
    radians * RAD_TO_DEG
}
#[cfg(feature = "std")]
pub fn sin(x: f32) -> f32 {
    x.sin()
}
#[cfg(not(feature = "std"))]
pub fn sin(x: f32) -> f32 {
    libm::sinf(x)
}
#[cfg(feature = "std")]
pub fn cos(x: f32) -> f32 {
    x.cos()
}
#[cfg(not(feature = "std"))]
pub fn cos(x: f32) -> f32 {
    libm::cosf(x)
}
#[cfg(feature = "std")]
pub fn tan(x: f32) -> f32 {
    x.tan()
}
#[cfg(not(feature = "std"))]
pub fn tan(x: f32) -> f32 {
    libm::tanf(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_returns_smaller_value() {
        assert_eq!(min(1.0, 2.0), 1.0);
        assert_eq!(min(2.0, 1.0), 1.0);
    }

    #[test]
    fn max_returns_larger_value() {
        assert_eq!(max(1.0, 2.0), 2.0);
        assert_eq!(max(2.0, 1.0), 2.0);
    }

    #[test]
    fn clamp_value_within_range() {
        assert_eq!(clamp(5.0, 1.0, 10.0), 5.0);
        assert_eq!(clamp(0.0, 1.0, 10.0), 1.0);
        assert_eq!(clamp(15.0, 1.0, 10.0), 10.0);
    }

    #[test]
    fn lerp_interpolates_correctly() {
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(10.0, 20.0, 0.25), 12.5);
    }

    #[test]
    fn approx_equal_within_epsilon() {
        assert!(approx_equal(1.0, 1.000001, EPSILON));
        assert!(!approx_equal(1.0, 1.1, EPSILON));
    }

    #[test]
    fn approx_zero_within_epsilon() {
        assert!(approx_zero(0.000001, EPSILON));
        assert!(!approx_zero(0.1, EPSILON));
    }

    #[test]
    fn degrees_to_radians_conversion() {
        assert_eq!(degrees_to_radians(180.0), PI);
        assert_eq!(degrees_to_radians(360.0), TAU);
    }

    #[test]
    fn radians_to_degrees_conversion() {
        assert_eq!(radians_to_degrees(PI), 180.0);
        assert_eq!(radians_to_degrees(TAU), 360.0);
    }

    #[test]
    fn sin_computes_correctly() {
        assert_eq!(sin(0.0), 0.0);
        assert!((sin(PI / 2.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn cos_computes_correctly() {
        assert_eq!(cos(0.0), 1.0);
        assert!((cos(PI) + 1.0).abs() < EPSILON);
    }

    #[test]
    fn tan_computes_correctly() {
        assert_eq!(tan(0.0), 0.0);
        assert!((tan(PI / 4.0) - 1.0).abs() < EPSILON);
    }
}
