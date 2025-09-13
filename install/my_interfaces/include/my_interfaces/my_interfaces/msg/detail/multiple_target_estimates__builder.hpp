// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/MultipleTargetEstimates.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/multiple_target_estimates__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_MultipleTargetEstimates_ground_truths
{
public:
  explicit Init_MultipleTargetEstimates_ground_truths(::my_interfaces::msg::MultipleTargetEstimates & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::MultipleTargetEstimates ground_truths(::my_interfaces::msg::MultipleTargetEstimates::_ground_truths_type arg)
  {
    msg_.ground_truths = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::MultipleTargetEstimates msg_;
};

class Init_MultipleTargetEstimates_target_estimates
{
public:
  explicit Init_MultipleTargetEstimates_target_estimates(::my_interfaces::msg::MultipleTargetEstimates & msg)
  : msg_(msg)
  {}
  Init_MultipleTargetEstimates_ground_truths target_estimates(::my_interfaces::msg::MultipleTargetEstimates::_target_estimates_type arg)
  {
    msg_.target_estimates = std::move(arg);
    return Init_MultipleTargetEstimates_ground_truths(msg_);
  }

private:
  ::my_interfaces::msg::MultipleTargetEstimates msg_;
};

class Init_MultipleTargetEstimates_num_of_targets
{
public:
  explicit Init_MultipleTargetEstimates_num_of_targets(::my_interfaces::msg::MultipleTargetEstimates & msg)
  : msg_(msg)
  {}
  Init_MultipleTargetEstimates_target_estimates num_of_targets(::my_interfaces::msg::MultipleTargetEstimates::_num_of_targets_type arg)
  {
    msg_.num_of_targets = std::move(arg);
    return Init_MultipleTargetEstimates_target_estimates(msg_);
  }

private:
  ::my_interfaces::msg::MultipleTargetEstimates msg_;
};

class Init_MultipleTargetEstimates_header
{
public:
  Init_MultipleTargetEstimates_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MultipleTargetEstimates_num_of_targets header(::my_interfaces::msg::MultipleTargetEstimates::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_MultipleTargetEstimates_num_of_targets(msg_);
  }

private:
  ::my_interfaces::msg::MultipleTargetEstimates msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::MultipleTargetEstimates>()
{
  return my_interfaces::msg::builder::Init_MultipleTargetEstimates_header();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__BUILDER_HPP_
