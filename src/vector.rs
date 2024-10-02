#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    pub fn add(&self, other: &Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn sub(&self, other: &Vec2) -> Vec2 {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Vec2 {
        Vec2 {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }

    pub fn dot(&self, other: &Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn unit(&self) -> Vec2 {
        let len = self.length();
        if len > 0.0 {
            self.mul_scalar(1.0 / len)
        } else {
            *self
        }
    }
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len > 0.0 {
            self.mul_scalar(1.0 / len)
        } else {
            *self
        }
    }
}

impl Vec4 {
    pub fn as_ptr(&self) -> *const f32 {
        &self.x as *const f32
    }
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4 { x, y, z, w }
    }

    pub fn add(&self, other: &Vec4) -> Vec4 {
        Vec4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }

    pub fn sub(&self, other: &Vec4) -> Vec4 {
        Vec4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Vec4 {
        Vec4 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }

    pub fn dot(&self, other: &Vec4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    #[cfg(target_arch = "x86_64")]
    pub fn dot_simd(&self, other: &Vec4) -> f32 {
        unsafe {
            let a = _mm_loadu_ps(self.as_ptr());
            let b = _mm_loadu_ps(other.as_ptr());
            let product = _mm_mul_ps(a, b);
            let sum1 = _mm_hadd_ps(product, product);
            let sum2 = _mm_hadd_ps(sum1, sum1);
            _mm_cvtss_f32(sum2)
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn dot_simd(&self, other: &Vec4) -> f32 {
        unsafe {
            let a = vld1q_f32(self.as_ptr());
            let b = vld1q_f32(other.as_ptr());
            let product = vmulq_f32(a, b);
            let sum1 = vaddq_f32(product, vextq_f32::<2>(product, product));
            let sum2 = vaddq_f32(sum1, vextq_f32::<1>(sum1, sum1));
            vgetq_lane_f32::<0>(sum2)
        }
    }
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn unit(&self) -> Vec4 {
        let len = self.length();
        if len > 0.0 {
            self.mul_scalar(1.0 / len)
        } else {
            *self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_creation() {
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vec2_add() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        let result = v1.add(&v2);
        assert_eq!(result, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn test_vec2_sub() {
        let v1 = Vec2::new(5.0, 6.0);
        let v2 = Vec2::new(2.0, 3.0);
        let result = v1.sub(&v2);
        assert_eq!(result, Vec2::new(3.0, 3.0));
    }

    #[test]
    fn test_vec2_mul_scalar() {
        let v = Vec2::new(1.0, 2.0);
        let result = v.mul_scalar(3.0);
        assert_eq!(result, Vec2::new(3.0, 6.0));
    }

    #[test]
    fn test_vec2_dot() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        let result = v1.dot(&v2);
        assert_eq!(result, 11.0);
    }

    #[test]
    fn test_vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        let result = v.length();
        assert_eq!(result, 5.0); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_vec2_unit() {
        let v = Vec2::new(3.0, 4.0);
        let unit_v = v.unit();
        assert_eq!(unit_v.length(), 1.0);
    }

    #[test]
    fn test_vec3_creation() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3_add() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let result = v1.add(&v2);
        assert_eq!(result, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vec3_sub() {
        let v1 = Vec3::new(7.0, 8.0, 9.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let result = v1.sub(&v2);
        assert_eq!(result, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vec3_mul_scalar() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = v.mul_scalar(2.0);
        assert_eq!(result, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vec3_dot() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let result = v1.dot(&v2);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_vec3_cross() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let result = v1.cross(&v2);
        assert_eq!(result, Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_vec3_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let result = v.length();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert_eq!(normalized.length(), 1.0);
    }

    #[test]
    fn test_vec4_creation() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4_add() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
        let result = v1.add(&v2);
        assert_eq!(result, Vec4::new(6.0, 8.0, 10.0, 12.0));
    }

    #[test]
    fn test_vec4_sub() {
        let v1 = Vec4::new(10.0, 11.0, 12.0, 13.0);
        let v2 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let result = v1.sub(&v2);
        assert_eq!(result, Vec4::new(9.0, 9.0, 9.0, 9.0));
    }

    #[test]
    fn test_vec4_mul_scalar() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let result = v.mul_scalar(2.0);
        assert_eq!(result, Vec4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_vec4_dot() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
        let result = v1.dot(&v2);
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_vec4_dot_simd() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
        let result = v1.dot_simd(&v2);
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_vec4_length() {
        let v = Vec4::new(1.0, 2.0, 2.0, 0.0);
        let result = v.length();
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_vec4_unit() {
        let v = Vec4::new(3.0, 4.0, 0.0, 0.0);
        let unit_v = v.unit();
        assert_eq!(unit_v.length(), 1.0);
    }
}
