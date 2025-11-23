// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/ObsAvoidanceDebug.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/obs_avoidance_debug__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_ObsAvoidanceDebug_u_safe
{
public:
  explicit Init_ObsAvoidanceDebug_u_safe(::my_interfaces::msg::ObsAvoidanceDebug & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::ObsAvoidanceDebug u_safe(::my_interfaces::msg::ObsAvoidanceDebug::_u_safe_type arg)
  {
    msg_.u_safe = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::ObsAvoidanceDebug msg_;
};

class Init_ObsAvoidanceDebug_beta
{
public:
  explicit Init_ObsAvoidanceDebug_beta(::my_interfaces::msg::ObsAvoidanceDebug & msg)
  : msg_(msg)
  {}
  Init_ObsAvoidanceDebug_u_safe beta(::my_interfaces::msg::ObsAvoidanceDebug::_beta_type arg)
  {
    msg_.beta = std::move(arg);
    return Init_ObsAvoidanceDebug_u_safe(msg_);
  }

private:
  ::my_interfaces::msg::ObsAvoidanceDebug msg_;
};

class Init_ObsAvoidanceDebug_alpha2_h
{
public:
  explicit Init_ObsAvoidanceDebug_alpha2_h(::my_interfaces::msg::ObsAvoidanceDebug & msg)
  : msg_(msg)
  {}
  Init_ObsAvoidanceDebug_beta alpha2_h(::my_interfaces::msg::ObsAvoidanceDebug::_alpha2_h_type arg)
  {
    msg_.alpha2_h = std::move(arg);
    return Init_ObsAvoidanceDebug_beta(msg_);
  }

private:
  ::my_interfaces::msg::ObsAvoidanceDebug msg_;
};

class Init_ObsAvoidanceDebug_two_alpha_h_hdot
{
public:
  explicit Init_ObsAvoidanceDebug_two_alpha_h_hdot(::my_interfaces::msg::ObsAvoidanceDebug & msg)
  : msg_(msg)
  {}
  Init_ObsAvoidanceDebug_alpha2_h two_alpha_h_hdot(::my_interfaces::msg::ObsAvoidanceDebug::_two_alpha_h_hdot_type arg)
  {
    msg_.two_alpha_h_hdot = std::move(arg);
    return Init_ObsAvoidanceDebug_alpha2_h(msg_);
  }

private:
  ::my_interfaces::msg::ObsAvoidanceDebug msg_;
};

class Init_ObsAvoidanceDebug_hddot
{
public:
  explicit Init_ObsAvoidanceDebug_hddot(::my_interfaces::msg::ObsAvoidanceDebug & msg)
  : msg_(msg)
  {}
  Init_ObsAvoidanceDebug_two_alpha_h_hdot hddot(::my_interfaces::msg::ObsAvoidanceDebug::_hddot_type arg)
  {
    msg_.hddot = std::move(arg);
    return Init_ObsAvoidanceDebug_two_alpha_h_hdot(msg_);
  }

private:
  ::my_interfaces::msg::ObsAvoidanceDebug msg_;
};

class Init_ObsAvoidanceDebug_psi
{
public:
  Init_ObsAvoidanceDebug_psi()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ObsAvoidanceDebug_hddot psi(::my_interfaces::msg::ObsAvoidanceDebug::_psi_type arg)
  {
    msg_.psi = std::move(arg);
    return Init_ObsAvoidanceDebug_hddot(msg_);
  }

private:
  ::my_interfaces::msg::ObsAvoidanceDebug msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::ObsAvoidanceDebug>()
{
  return my_interfaces::msg::builder::Init_ObsAvoidanceDebug_psi();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__BUILDER_HPP_
