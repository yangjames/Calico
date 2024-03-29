#ifndef CALICO_STATUS_BUILDER_H_
#define CALICO_STATUS_BUILDER_H_

#include <sstream>
#include <string>
#include <utility>
 
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
 
namespace util {
 
class StatusBuilder {
 public:
  explicit StatusBuilder(const absl::StatusCode code)
      : base_status_(code, /*msg=*/{}) {}
 
  explicit StatusBuilder(const absl::Status status)
      : base_status_(std::move(status)) {}
 
  operator absl::Status() const {  // NOLINT
    const std::string annotation = ss_.str();
    if (annotation.empty()) {
      return base_status_;
    }
    if (base_status_.message().empty()) {
      return absl::Status(base_status_.code(), annotation);
    }
    const std::string annotated_message =
        absl::StrCat(base_status_.message(), "; ", annotation);
    return absl::Status(base_status_.code(), annotated_message);
  }
 
  template <class T>
  StatusBuilder& operator<<(const T& t) {
    ss_ << t;
    return *this;
  }
 
  StatusBuilder& SetAppend() { return *this; }
 
 private:
  const absl::Status base_status_;
  std::ostringstream ss_;
};
 
inline StatusBuilder AbortedErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kAborted);
}
 
inline StatusBuilder AlreadyExistsErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kAlreadyExists);
}
 
inline StatusBuilder CancelledErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kCancelled);
}
 
inline StatusBuilder DataLossErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kDataLoss);
}
 
inline StatusBuilder DeadlineExceededErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kDeadlineExceeded);
}
 
inline StatusBuilder FailedPreconditionErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kFailedPrecondition);
}
 
inline StatusBuilder InternalErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kInternal);
}
 
inline StatusBuilder InvalidArgumentErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kInvalidArgument);
}
 
inline StatusBuilder NotFoundErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kNotFound);
}
 
inline StatusBuilder OutOfRangeErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kOutOfRange);
}
 
inline StatusBuilder PermissionDeniedErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kPermissionDenied);
}
 
inline StatusBuilder UnauthenticatedErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kUnauthenticated);
}
 
inline StatusBuilder ResourceExhaustedErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kResourceExhausted);
}
 
inline StatusBuilder UnavailableErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kUnavailable);
}
 
inline StatusBuilder UnimplementedErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kUnimplemented);
}
 
inline StatusBuilder UnknownErrorBuilder() {
  return StatusBuilder(absl::StatusCode::kUnknown);
}
 
}  // namespace util
 
#endif  // CALICO_STATUS_BUILDER_H_
