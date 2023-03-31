#ifndef CALICO_STATUSOR_MACROS_H_
#define CALICO_STATUSOR_MACROS_H_


#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "calico/status_builder.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


// absl matchers pulled/derived from here:
// https://github.com/abseil/abseil-cpp/issues/951#issuecomment-828460483
inline const ::absl::Status& GetStatus(const ::absl::Status& status) {
  return status;
}

template <typename T>
inline const ::absl::Status& GetStatus(const ::absl::StatusOr<T>& status) {
  return status.status();
}

// Monomorphic implementation of matcher IsOkAndHolds(m).  StatusOrType is a
// reference to StatusOr<T>.
template <typename StatusOrType>
class IsOkAndHoldsMatcherImpl
    : public ::testing::MatcherInterface<StatusOrType> {
 public:
  typedef
      typename std::remove_reference<StatusOrType>::type::value_type value_type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const value_type&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      StatusOrType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    if (!actual_value.ok()) {
      *result_listener << "which has status " << actual_value.status();
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    const bool matches =
        inner_matcher_.MatchAndExplain(*actual_value, &inner_listener);
    const std::string inner_explanation = inner_listener.str();
    if (!inner_explanation.empty()) {
      *result_listener << "which contains value "
                       << ::testing::PrintToString(*actual_value) << ", "
                       << inner_explanation;
    }
    return matches;
  }

 private:
  const ::testing::Matcher<const value_type&> inner_matcher_;
};

// Implements IsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type.  StatusOrType can be either StatusOr<T> or a
  // reference to StatusOr<T>.
  template <typename StatusOrType>
  operator ::testing::Matcher<StatusOrType>() const {  // NOLINT
    return ::testing::Matcher<StatusOrType>(
        new IsOkAndHoldsMatcherImpl<const StatusOrType&>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  void DescribeTo(std::ostream* os) const override { *os << "is OK"; }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not OK";
  }
  bool MatchAndExplain(T actual_value,
                       ::testing::MatchResultListener*) const override {
    return GetStatus(actual_value).ok();
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::Matcher<T>(new MonoIsOkMatcherImpl<T>());
  }
};

// Returns a gMock matcher that matches a StatusOr<> whose status is
// OK and whose value matches the inner matcher.
template <typename InnerMatcher>
IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type> IsOkAndHolds(
    InnerMatcher&& inner_matcher) {
  return IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

// Returns a gMock matcher that matches a Status or StatusOr<> which is OK.
inline IsOkMatcher IsOk() { return IsOkMatcher(); }

// Macros for testing the results of functions that return absl::Status or
// absl::StatusOr<T> (for any type T).
//#define EXPECT_OK(expression) EXPECT_THAT(expression, IsOk())

// Macros for testing the results of functions that returns absl::Status.
#define EXPECT_OK(statement) \
  EXPECT_EQ(absl::OkStatus(), (statement))
#define ASSERT_OK(statement) \
  ASSERT_EQ(absl::OkStatus(), (statement))

// Run a command that returns a absl::Status.  If the called code returns an
// error status, return that status up out of this method too.
//
// Example:
//   RETURN_IF_ERROR(DoThings(4));
//   RETURN_IF_ERROR(DoThings(5)) << "Additional error context";
#define RETURN_IF_ERROR(expr)                                        \
  switch (0)                                                         \
  case 0:                                                            \
  default:                                                           \
    if (const ::absl::Status status_macro_internal_adaptor = (expr); \
        status_macro_internal_adaptor.ok()) {                        \
    } else /* NOLINT */                                              \
      return ::util::StatusBuilder(status_macro_internal_adaptor)
 
// Executes an expression that returns an absl::StatusOr, extracting its value
// into the variable defined by lhs (or returning on error).
//
// Example: Assigning to an existing value
//   ValueType value;
//   ASSIGN_OR_RETURN(value, MaybeGetValue(arg));
//   ASSIGN_OR_RETURN((auto [key, val]), MaybeGetValue(arg));
//
// WARNING: ASSIGN_OR_RETURN expands into multiple statements; it cannot be used
//  in a single statement (e.g. as the body of an if statement without {})!
#define ASSIGN_OR_RETURN(lhs, rexpr)    \
  STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_( \
      STATUS_MACROS_IMPL_CONCAT_(_status_or_value, __COUNTER__), lhs, rexpr);
 
#define STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                         \
  RETURN_IF_ERROR(statusor.status());                              \
  STATUS_MACROS_IMPL_UNPARENTHESIS(lhs) = std::move(statusor).value()
 
// Internal helpers for macro expansion.
#define STATUS_MACROS_IMPL_UNPARENTHESIS_INNER(...) \
  STATUS_MACROS_IMPL_UNPARENTHESIS_INNER_(__VA_ARGS__)
#define STATUS_MACROS_IMPL_UNPARENTHESIS_INNER_(...) \
  STATUS_MACROS_IMPL_VAN##__VA_ARGS__
#define ISH(...) ISH __VA_ARGS__
#define STATUS_MACROS_IMPL_VANISH
 
// If the input is parenthesized, removes the parentheses. Otherwise expands to
// the input unchanged.
#define STATUS_MACROS_IMPL_UNPARENTHESIS(...) \
  STATUS_MACROS_IMPL_UNPARENTHESIS_INNER(ISH __VA_ARGS__)
 
// Internal helper for concatenating macro values.
#define STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define STATUS_MACROS_IMPL_CONCAT_(x, y) STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)

#define CONCAT_IMPL(x, y) x##y
#define CONCAT_MACRO(x, y) CONCAT_IMPL(x, y)
#define ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)  \
  auto statusor = (rexpr);                               \
  ASSERT_TRUE(statusor.status().ok()) <<                 \
      statusor.status();                                 \
  lhs = std::move(statusor.value())

#define ASSERT_OK_AND_ASSIGN(lhs, rexpr)        \
  ASSERT_OK_AND_ASSIGN_IMPL(CONCAT_MACRO(       \
      _status_or, __COUNTER__), lhs, rexpr)

#endif // CALICO_STATUSOR_MACROS_H_
