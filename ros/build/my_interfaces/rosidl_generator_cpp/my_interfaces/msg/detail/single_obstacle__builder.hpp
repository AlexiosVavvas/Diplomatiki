// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/SingleObstacle.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/single_obstacle__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_SingleObstacle_rho0
{
public:
  explicit Init_SingleObstacle_rho0(::my_interfaces::msg::SingleObstacle & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::SingleObstacle rho0(::my_interfaces::msg::SingleObstacle::_rho0_type arg)
  {
    msg_.rho0 = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::SingleObstacle msg_;
};

class Init_SingleObstacle_kappa
{
public:
  explicit Init_SingleObstacle_kappa(::my_interfaces::msg::SingleObstacle & msg)
  : msg_(msg)
  {}
  Init_SingleObstacle_rho0 kappa(::my_interfaces::msg::SingleObstacle::_kappa_type arg)
  {
    msg_.kappa = std::move(arg);
    return Init_SingleObstacle_rho0(msg_);
  }

private:
  ::my_interfaces::msg::SingleObstacle msg_;
};

class Init_SingleObstacle_dimensions
{
public:
  explicit Init_SingleObstacle_dimensions(::my_interfaces::msg::SingleObstacle & msg)
  : msg_(msg)
  {}
  Init_SingleObstacle_kappa dimensions(::my_interfaces::msg::SingleObstacle::_dimensions_type arg)
  {
    msg_.dimensions = std::move(arg);
    return Init_SingleObstacle_kappa(msg_);
  }

private:
  ::my_interfaces::msg::SingleObstacle msg_;
};

class Init_SingleObstacle_position
{
public:
  explicit Init_SingleObstacle_position(::my_interfaces::msg::SingleObstacle & msg)
  : msg_(msg)
  {}
  Init_SingleObstacle_dimensions position(::my_interfaces::msg::SingleObstacle::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_SingleObstacle_dimensions(msg_);
  }

private:
  ::my_interfaces::msg::SingleObstacle msg_;
};

class Init_SingleObstacle_obs_name
{
public:
  explicit Init_SingleObstacle_obs_name(::my_interfaces::msg::SingleObstacle & msg)
  : msg_(msg)
  {}
  Init_SingleObstacle_position obs_name(::my_interfaces::msg::SingleObstacle::_obs_name_type arg)
  {
    msg_.obs_name = std::move(arg);
    return Init_SingleObstacle_position(msg_);
  }

private:
  ::my_interfaces::msg::SingleObstacle msg_;
};

class Init_SingleObstacle_obs_type
{
public:
  Init_SingleObstacle_obs_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SingleObstacle_obs_name obs_type(::my_interfaces::msg::SingleObstacle::_obs_type_type arg)
  {
    msg_.obs_type = std::move(arg);
    return Init_SingleObstacle_obs_name(msg_);
  }

private:
  ::my_interfaces::msg::SingleObstacle msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::SingleObstacle>()
{
  return my_interfaces::msg::builder::Init_SingleObstacle_obs_type();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__BUILDER_HPP_
