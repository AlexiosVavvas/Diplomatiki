// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/JoystickData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/joystick_data__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_JoystickData_switch_state
{
public:
  explicit Init_JoystickData_switch_state(::my_interfaces::msg::JoystickData & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::JoystickData switch_state(::my_interfaces::msg::JoystickData::_switch_state_type arg)
  {
    msg_.switch_state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::JoystickData msg_;
};

class Init_JoystickData_rudder
{
public:
  explicit Init_JoystickData_rudder(::my_interfaces::msg::JoystickData & msg)
  : msg_(msg)
  {}
  Init_JoystickData_switch_state rudder(::my_interfaces::msg::JoystickData::_rudder_type arg)
  {
    msg_.rudder = std::move(arg);
    return Init_JoystickData_switch_state(msg_);
  }

private:
  ::my_interfaces::msg::JoystickData msg_;
};

class Init_JoystickData_elevator
{
public:
  explicit Init_JoystickData_elevator(::my_interfaces::msg::JoystickData & msg)
  : msg_(msg)
  {}
  Init_JoystickData_rudder elevator(::my_interfaces::msg::JoystickData::_elevator_type arg)
  {
    msg_.elevator = std::move(arg);
    return Init_JoystickData_rudder(msg_);
  }

private:
  ::my_interfaces::msg::JoystickData msg_;
};

class Init_JoystickData_aileron
{
public:
  explicit Init_JoystickData_aileron(::my_interfaces::msg::JoystickData & msg)
  : msg_(msg)
  {}
  Init_JoystickData_elevator aileron(::my_interfaces::msg::JoystickData::_aileron_type arg)
  {
    msg_.aileron = std::move(arg);
    return Init_JoystickData_elevator(msg_);
  }

private:
  ::my_interfaces::msg::JoystickData msg_;
};

class Init_JoystickData_throttle
{
public:
  Init_JoystickData_throttle()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JoystickData_aileron throttle(::my_interfaces::msg::JoystickData::_throttle_type arg)
  {
    msg_.throttle = std::move(arg);
    return Init_JoystickData_aileron(msg_);
  }

private:
  ::my_interfaces::msg::JoystickData msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::JoystickData>()
{
  return my_interfaces::msg::builder::Init_JoystickData_throttle();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__BUILDER_HPP_
