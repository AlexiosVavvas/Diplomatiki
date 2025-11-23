// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/JoystickData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/joystick_data__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const JoystickData & msg,
  std::ostream & out)
{
  out << "{";
  // member: throttle
  {
    out << "throttle: ";
    rosidl_generator_traits::value_to_yaml(msg.throttle, out);
    out << ", ";
  }

  // member: aileron
  {
    out << "aileron: ";
    rosidl_generator_traits::value_to_yaml(msg.aileron, out);
    out << ", ";
  }

  // member: elevator
  {
    out << "elevator: ";
    rosidl_generator_traits::value_to_yaml(msg.elevator, out);
    out << ", ";
  }

  // member: rudder
  {
    out << "rudder: ";
    rosidl_generator_traits::value_to_yaml(msg.rudder, out);
    out << ", ";
  }

  // member: switch_state
  {
    out << "switch_state: ";
    rosidl_generator_traits::value_to_yaml(msg.switch_state, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const JoystickData & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: throttle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "throttle: ";
    rosidl_generator_traits::value_to_yaml(msg.throttle, out);
    out << "\n";
  }

  // member: aileron
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "aileron: ";
    rosidl_generator_traits::value_to_yaml(msg.aileron, out);
    out << "\n";
  }

  // member: elevator
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "elevator: ";
    rosidl_generator_traits::value_to_yaml(msg.elevator, out);
    out << "\n";
  }

  // member: rudder
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rudder: ";
    rosidl_generator_traits::value_to_yaml(msg.rudder, out);
    out << "\n";
  }

  // member: switch_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "switch_state: ";
    rosidl_generator_traits::value_to_yaml(msg.switch_state, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const JoystickData & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace my_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use my_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const my_interfaces::msg::JoystickData & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::JoystickData & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::JoystickData>()
{
  return "my_interfaces::msg::JoystickData";
}

template<>
inline const char * name<my_interfaces::msg::JoystickData>()
{
  return "my_interfaces/msg/JoystickData";
}

template<>
struct has_fixed_size<my_interfaces::msg::JoystickData>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<my_interfaces::msg::JoystickData>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<my_interfaces::msg::JoystickData>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__TRAITS_HPP_
