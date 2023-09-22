#ifndef CALICO_SENSORS_GYROSCOPE_MODELS_H_
#define CALICO_SENSORS_GYROSCOPE_MODELS_H_

#include "calico/typedefs.h"

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "Eigen/Dense"


namespace calico::sensors {

/// Gyroscope model types.
enum class GyroscopeIntrinsicsModel : int {
  /// Default no model.
  kNone,
  /// Isotropic scale without bias.
  kGyroscopeScaleOnly,
  /// Isotropic scale with bias.
  kGyroscopeScaleAndBias,
  /// VectorNav model.
  kGyroscopeVectorNav,
};

/// Base class for gyroscope models.
class GyroscopeModel {
 public:
  virtual ~GyroscopeModel() = default;

  /// Project an angular velocity vector through the intrinsics model.
  /// Top level call invokes the derived class's implementation.
  template <typename T>
  absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& omega_sensor_world) const;

  /// Invert the intrinsics model to get an angular velocity vector.
  /// Top level call invokes the derived class's implementation.
  template <typename T>
  absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) const;

  /// Getter for gyroscope model type.
  virtual GyroscopeIntrinsicsModel GetType() const  = 0;

  /// Getter for the number of parameters for this model.
  virtual int NumberOfParameters() const = 0;

  // TODO(yangjames): This method should return an absl::StatusOr. Figure out
  //   how to support this using unique_ptr's, or find macros that already
  //   implement this feature (i.e. ASSIGN_OR_RETURN).
  /// Factory method for creating a gyroscope model with `gyroscope_model` type.
  /// This method will return a nullptr if an unsupported
  /// GyroscopeIntrinsicsModel is passed in.
  static std::unique_ptr<GyroscopeModel> Create(
      GyroscopeIntrinsicsModel gyroscope_model);
};


/// 1-parameter isotropic scale intrinsics model.
/// \f$[s]\f$
class GyroscopeScaleOnlyModel : public GyroscopeModel {
 public:
  static constexpr int kNumberOfParameters = 1;
  static constexpr GyroscopeIntrinsicsModel kModelType = GyroscopeIntrinsicsModel::kGyroscopeScaleOnly;

  GyroscopeScaleOnlyModel() = default;
  ~GyroscopeScaleOnlyModel() override = default;
  GyroscopeScaleOnlyModel& operator=(const GyroscopeScaleOnlyModel&) = default;

  /// Returns measurement \f$\mathbf{f}\f$, a 3-D vector such that
  /// \f[
  /// \mathbf{f} = s\boldsymbol{\omega}^s_{ws}
  /// \f]\n\n
  /// `omega_sensor_world` is the angular velocity of the sensor relative to
  /// the inertial world frame resolved in the sensor frame
  /// \f$\boldsymbol{\omega}^s_{ws}\f$.
  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& omega_sensor_world) {
    const T& scale = intrinsics(0);
    return scale * omega_sensor_world;
  }

  /// Inverts the measurement model to obtain angular rate as observed by the
  /// sensor.
  /// \f[
  /// \boldsymbol{\omega}^s_{ws} = \frac{1}{s}\mathbf{f}
  /// \f]
  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) {
    const T& inv_scale = static_cast<T>(1.0) / intrinsics(0);
    return inv_scale * measurement;
  }

  GyroscopeIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};

/// 4-parameter scale + bias intrinsics model.
/// Parameter order:
///   \f$[s, b_x, b_y, b_z]\f$
class GyroscopeScaleAndBiasModel : public GyroscopeModel {
 public:
  static constexpr int kNumberOfParameters = 4;
  static constexpr GyroscopeIntrinsicsModel kModelType = GyroscopeIntrinsicsModel::kGyroscopeScaleAndBias;

  GyroscopeScaleAndBiasModel() = default;
  ~GyroscopeScaleAndBiasModel() override = default;
  GyroscopeScaleAndBiasModel& operator=(
      const GyroscopeScaleAndBiasModel&) = default;

  /// Returns measurement \f$\mathbf{f}\f$, a 3-D vector such that
  /// \f[
  /// \mathbf{f} = s\boldsymbol{\omega}^s_{ws} +
  /// \left[\begin{matrix}b_x\\b_y\\b_z\end{matrix}\right]
  /// \f]
  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& omega_sensor_world) {
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid gyroscope intrinsics size. Expected ", kNumberOfParameters,
          ", got ", intrinsics.size()));
    }
    const T& scale = intrinsics(0);
    const Eigen::Ref<const Eigen::Vector3<T>> bias =
        intrinsics.segment(1, 3);
    return scale * omega_sensor_world + bias;
  }

  /// Inverts the measurement model to obtain angular rate as observed by the
  /// sensor.
  /// \f[
  /// \boldsymbol{\omega}^s_{ws} = \frac{1}{s}\left(\mathbf{f} -
  /// \left[\begin{matrix}b_x\\b_y\\b_z\end{matrix}\right]\right)
  /// \f]
  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) {
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid gyroscope intrinsics size. Expected ", kNumberOfParameters,
          ", got ", intrinsics.size()));
    }
    const T& inv_scale = static_cast<T>(1.0) / intrinsics(0);
    const Eigen::Ref<const Eigen::Vector3<T>> bias =
        intrinsics.segment(1, 3);
    return inv_scale * (measurement - bias);
  }

  GyroscopeIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};


/// Vector-Nav model. See [their page]
/// (https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-imucal)
/// for more details.\n
/// Parameter order:
///   \f$[s_x, s_y, s_z, a_1, a_2, a_3, a_4, a_5, a_6, b_x, b_y, b_z]\f$
class GyroscopeVectorNavModel : public GyroscopeModel {
 public:
  static constexpr int kNumberOfParameters = 12;
  static constexpr GyroscopeIntrinsicsModel kModelType = GyroscopeIntrinsicsModel::kGyroscopeVectorNav;

  GyroscopeVectorNavModel() = default;
  ~GyroscopeVectorNavModel() override = default;
  GyroscopeVectorNavModel& operator=(
      const GyroscopeVectorNavModel&) = default;

  /// Returns measurement \f$\mathbf{f}\f$, a 3-D vector such that
  /// \f[
  /// \mathbf{f} =
  ///   \mathbf{S}\mathbf{A}\boldsymbol{\omega}^s_{ws} + \mathbf{b}\\\\
  /// 
  /// \mathbf{S} = \left[\begin{matrix}
  ///     s_x & 0 & 0\\
  ///     0 & s_y & 0\\
  ///     0 & 0 & s_z
  ///   \end{matrix}\right],
  /// \mathbf{A} = \left[\begin{matrix}
  ///     1 & a_1 & a_2\\
  ///     a_3 & 1 & a_4\\
  ///     a_5 & a_6 & 1
  ///   \end{matrix}\right],
  /// \mathbf{b} = \left[\begin{matrix}b_x\\b_y\\b_z\end{matrix}\right]
  /// \f]
  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& omega_sensor_world) {
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid gyroscope intrinsics size. Expected ", kNumberOfParameters,
          ", got ", intrinsics.size()));
    }
    const T& sx = intrinsics(0);
    const T& sy = intrinsics(1);
    const T& sz = intrinsics(2);
    const T& a1 = intrinsics(3);
    const T& a2 = intrinsics(4);
    const T& a3 = intrinsics(5);
    const T& a4 = intrinsics(6);
    const T& a5 = intrinsics(7);
    const T& a6 = intrinsics(8);
    const T& bx = intrinsics(9);
    const T& by = intrinsics(10);
    const T& bz = intrinsics(11);
    const T& wx = omega_sensor_world.x();
    const T& wy = omega_sensor_world.y();
    const T& wz = omega_sensor_world.z();
    const T fx = bx + sx * (wx + a1 * wy + a2 * wz);
    const T fy = by + sy * (wy + a3 * wx + a4 * wz);
    const T fz = bz + sz * (wz + a5 * wx + a6 * wy);
    return Eigen::Vector3<T>(fx, fy, fz);
  }

  /// Inverts the measurement model to obtain angular rate as observed by the
  /// sensor.
  /// \f[
  /// \boldsymbol{\omega}^s_{ws} =
  ///   \left(\mathbf{S}\mathbf{A}\right)^{-1}
  ///   \left(\mathbf{f} - \mathbf{b}\right)\\\\
  /// \mathbf{S} = \left[\begin{matrix}
  ///     s_x & 0 & 0\\
  ///     0 & s_y & 0\\
  ///     0 & 0 & s_z
  ///   \end{matrix}\right], 
  /// \mathbf{A} = \left[\begin{matrix}
  ///     1 & a_1 & a_2\\
  ///     a_3 & 1 & a_4\\
  ///     a_5 & a_6 & 1
  ///   \end{matrix}\right],
  /// \mathbf{b} = \left[\begin{matrix}b_x\\b_y\\b_z\end{matrix}\right]
  /// \f]
  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) {
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid gyroscope intrinsics size. Expected ", kNumberOfParameters,
          ", got ", intrinsics.size()));
    }
    const T& sx = intrinsics(0);
    const T& sy = intrinsics(1);
    const T& sz = intrinsics(2);
    const T& a1 = intrinsics(3);
    const T& a2 = intrinsics(4);
    const T& a3 = intrinsics(5);
    const T& a4 = intrinsics(6);
    const T& a5 = intrinsics(7);
    const T& a6 = intrinsics(8);
    const T& fx = measurement.x();
    const T& fy = measurement.y();
    const T& fz = measurement.z();
    const Eigen::Vector3<T> bias(intrinsics(9), intrinsics(10), intrinsics(11));
    const Eigen::Vector3<T> d = bias - measurement;
    const T det_A =
      1.0 - a1*a3 - a2*a5 - a4*a6 + a1*a4*a5 + a2*a3*a6;
    const T dx_inv_sx_detA = d.x() / (sx * det_A);
    const T dy_inv_sy_detA = d.y() / (sy * det_A);
    const T dz_inv_sz_detA = d.z() / (sz * det_A);
    const T wx = (a4*a6 - 1) * dx_inv_sx_detA + (a1 - a2*a6) * dy_inv_sy_detA +
        (a2 - a1*a4) * dz_inv_sz_detA;
    const T wy = (a2*a5 - 1) * dy_inv_sy_detA + (a3 - a4*a5) * dx_inv_sx_detA +
        (a4 - a2*a3) * dz_inv_sz_detA;
    const T wz = (a1*a3 - 1) * dz_inv_sz_detA + (a5 - a3*a6) * dx_inv_sx_detA +
        (a6 - a1*a5) * dy_inv_sy_detA;
    return Eigen::Vector3<T>(wx, wy, wz);
  }

  GyroscopeIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};



template <typename T>
absl::StatusOr<Eigen::Vector3<T>> GyroscopeModel::Project(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector3<T>& omega_sensor_world) const {
  if (const auto derived = dynamic_cast<const GyroscopeScaleOnlyModel*>(this)) {
    return derived->Project(intrinsics, omega_sensor_world);
  }
  if (const auto derived =
          dynamic_cast<const GyroscopeScaleAndBiasModel*>(this)) {
    return derived->Project(intrinsics, omega_sensor_world);
  }
  if (const auto derived =
          dynamic_cast<const GyroscopeVectorNavModel*>(this)) {
    return derived->Project(intrinsics, omega_sensor_world);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Project for gyroscope model ", this->GetType(), " not supported."));
}

template <typename T>
absl::StatusOr<Eigen::Vector3<T>> GyroscopeModel::Unproject(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector3<T>& measurement) const {
  if (const auto derived = dynamic_cast<const GyroscopeScaleOnlyModel*>(this)) {
    return derived->Unproject(intrinsics, measurement);
  }
  if (const auto derived =
          dynamic_cast<const GyroscopeScaleAndBiasModel*>(this)) {
    return derived->Unproject(intrinsics, measurement);
  }
  if (const auto derived =
          dynamic_cast<const GyroscopeVectorNavModel*>(this)) {
    return derived->Unproject(intrinsics, measurement);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unproject for gyroscope model ", this->GetType(), " not supported."));
}

} // namespace calico::sensors

#endif // CALICO_SENSORS_GYROSCOPE_MODELS_H_
