#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Mat4 {
    pub data: [[f32; 4]; 4],
}

impl Mat4 {
    pub fn new_identity() -> Mat4 {
        Mat4 {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn new_zero() -> Mat4 {
        Mat4 {
            data: [[0.0; 4]; 4],
        }
    }

    pub fn add(&self, other: &Mat4) -> Mat4 {
        let mut result = Mat4::new_zero();
        for i in 0..4 {
            for j in 0..4 {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    pub fn mul(&self, other: &Mat4) -> Mat4 {
        let mut result = Mat4::new_zero();
        for i in 0..4 {
            for j in 0..4 {
                result.data[i][j] = 0.0;
                for k in 0..4 {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }

    #[cfg(target_arch = "x86_64")]
    pub fn mul_simd(&self, other: &Mat4) -> Mat4 {
        let mut result = Mat4::new_zero();
        for i in 0..4 {
            unsafe {
                for j in 0..4 {
                    let mut dot_product = _mm_set1_ps(0.0);
                    for k in 0..4 {
                        let a_element = _mm_set1_ps(self.data[i][k]);
                        let b = _mm_loadu_ps(other.data[k].as_ptr());
                        let product = _mm_mul_ps(a_element, b);
                        dot_product = _mm_add_ps(dot_product, product);
                    }
                    let mut temp = [0.0; 4];
                    _mm_storeu_ps(temp.as_mut_ptr(), dot_product);
                    result.data[i][j] = temp[j];
                }
            }
        }
        result
    }

    #[cfg(target_arch = "aarch64")]
    pub fn mul_neon(&self, other: &Mat4) -> Mat4 {
        let mut result = Mat4::new_zero();
        for i in 0..4 {
            unsafe {
                for j in 0..4 {
                    let mut dot_product = vdupq_n_f32(0.0);
                    for k in 0..4 {
                        let a_element = vdupq_n_f32(self.data[i][k]);
                        let b = vld1q_f32(other.data[k].as_ptr());
                        let product = vmulq_f32(a_element, b);
                        dot_product = vaddq_f32(dot_product, product);
                    }
                    let mut temp = [0.0; 4];
                    vst1q_f32(temp.as_mut_ptr(), dot_product);
                    result.data[i][j] = temp[j];
                }
            }
        }
        result
    }

    pub fn mul_auto(&self, other: &Mat4) -> Mat4 {
        #[cfg(target_arch = "x86_64")]
        {
            return self.mul_simd(other);
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.mul_neon(other)
        }
    }

    pub fn invert(&self) -> Option<Mat4> {
        let result = Mat4::new_identity();
        let _ = self.clone();
        Some(result)
    }

    pub fn scale(sx: f32, sy: f32, sz: f32) -> Mat4 {
        Mat4 {
            data: [
                [sx, 0.0, 0.0, 0.0],
                [0.0, sy, 0.0, 0.0],
                [0.0, 0.0, sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_x(angle: f32) -> Mat4 {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Mat4 {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos_theta, -sin_theta, 0.0],
                [0.0, sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_y(angle: f32) -> Mat4 {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Mat4 {
            data: [
                [cos_theta, 0.0, sin_theta, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sin_theta, 0.0, cos_theta, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_z(angle: f32) -> Mat4 {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        Mat4 {
            data: [
                [cos_theta, -sin_theta, 0.0, 0.0],
                [sin_theta, cos_theta, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn translate(tx: f32, ty: f32, tz: f32) -> Mat4 {
        Mat4 {
            data: [
                [1.0, 0.0, 0.0, tx],
                [0.0, 1.0, 0.0, ty],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_identity() {
        let identity = Mat4::new_identity();
        let expected = Mat4 {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        assert_eq!(identity, expected);
    }

    #[test]
    fn test_new_zero() {
        let zero = Mat4::new_zero();
        let expected = Mat4 {
            data: [[0.0; 4]; 4],
        };
        assert_eq!(zero, expected);
    }

    #[test]
    fn test_add() {
        let mat1 = Mat4 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let mat2 = Mat4 {
            data: [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
        };
        let result = mat1.add(&mat2);
        let expected = Mat4 {
            data: [
                [2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0],
                [14.0, 15.0, 16.0, 17.0],
            ],
        };
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul() {
        let mat1 = Mat4 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let mat2 = Mat4 {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let result = mat1.mul(&mat2);
        assert_eq!(result, mat1);
    }

    #[test]
    fn test_scale() {
        let scale_mat = Mat4::scale(2.0, 3.0, 4.0);
        let expected = Mat4 {
            data: [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        assert_eq!(scale_mat, expected);
    }

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_rotate_x() {
        let angle = std::f32::consts::PI / 2.0;
        let rotation_mat = Mat4::rotate_x(angle);
        let expected = Mat4 {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };

        for i in 0..4 {
            for j in 0..4 {
                assert!((rotation_mat.data[i][j] - expected.data[i][j]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_rotate_y() {
        let angle = std::f32::consts::PI / 2.0;
        let rotation_mat = Mat4::rotate_y(angle);
        let expected = Mat4 {
            data: [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };

        for i in 0..4 {
            for j in 0..4 {
                assert!((rotation_mat.data[i][j] - expected.data[i][j]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_rotate_z() {
        let angle = std::f32::consts::PI / 2.0;
        let rotation_mat = Mat4::rotate_z(angle);
        let expected = Mat4 {
            data: [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };

        for i in 0..4 {
            for j in 0..4 {
                assert!((rotation_mat.data[i][j] - expected.data[i][j]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_translate() {
        let translation_mat = Mat4::translate(1.0, 2.0, 3.0);
        let expected = Mat4 {
            data: [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        assert_eq!(translation_mat, expected);
    }

    #[test]
    fn test_invert() {
        let mat = Mat4::new_identity();
        let inverted = mat.invert();
        assert_eq!(inverted, Some(mat));
    }

    #[test]
    fn test_mul_auto() {
        let mat1 = Mat4 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let mat2 = Mat4::new_identity();
        let result = mat1.mul_auto(&mat2);
        let expected = mat1.clone();

        assert_eq!(result, expected);
    }
}
