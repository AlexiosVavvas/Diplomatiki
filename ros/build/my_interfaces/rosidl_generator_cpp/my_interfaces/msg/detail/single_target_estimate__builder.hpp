// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/SingleTargetEstimate.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/single_target_estimate__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_SingleTargetEstimate_covariance
{
public:
  explicit Init_SingleTargetEstimate_covariance(::my_interfaces::msg::SingleTargetEstimate & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::SingleTargetEstimate covariance(::my_interfaces::msg::SingleTargetEstimate::_covariance_type arg)
  {
    msg_.covariance = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::SingleTargetEstimate msg_;
};

class Init_SingleTargetEstimate_position
{
public:
  explicit Init_SingleTargetEstimate_position(::my_interfaces::msg::SingleTargetEstimate & msg)
  : msg_(msg)
  {}
  Init_SingleTargetEstimate_covariance position(::my_interfaces::msg::SingleTargetEstimate::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_SingleTargetEstimate_covariance(msg_);
  }

private:
  ::my_interfaces::msg::SingleTargetEstimate msg_;
};

class Init_SingleTargetEstimate_target_id
{
public:
  Init_SingleTargetEstimate_target_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SingleTargetEstimate_position target_id(::my_interfaces::msg::SingleTargetEstimate::_target_id_type arg)
  {
    msg_.target_id = std::move(arg);
    return Init_SingleTargetEstimate_position(msg_);
  }

private:
  ::my_interfaces::msg::SingleTargetEstimate msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::SingleTargetEstimate>()
{
  return my_interfaces::msg::builder::Init_SingleTargetEstimate_target_id();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__BUILDER_HPP_
