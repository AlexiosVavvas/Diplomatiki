// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/MultipleObstacles.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/multiple_obstacles__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'obstacles'
#include "my_interfaces/msg/detail/single_obstacle__traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const MultipleObstacles & msg,
  std::ostream & out)
{
  out << "{";
  // member: num_of_obstacles
  {
    out << "num_of_obstacles: ";
    rosidl_generator_traits::value_to_yaml(msg.num_of_obstacles, out);
    out << ", ";
  }

  // member: obstacles
  {
    if (msg.obstacles.size() == 0) {
      out << "obstacles: []";
    } else {
      out << "obstacles: [";
      size_t pending_items = msg.obstacles.size();
      for (auto item : msg.obstacles) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MultipleObstacles & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: num_of_obstacles
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_of_obstacles: ";
    rosidl_generator_traits::value_to_yaml(msg.num_of_obstacles, out);
    out << "\n";
  }

  // member: obstacles
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.obstacles.size() == 0) {
      out << "obstacles: []\n";
    } else {
      out << "obstacles:\n";
      for (auto item : msg.obstacles) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MultipleObstacles & msg, bool use_flow_style = false)
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
  const my_interfaces::msg::MultipleObstacles & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::MultipleObstacles & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::MultipleObstacles>()
{
  return "my_interfaces::msg::MultipleObstacles";
}

template<>
inline const char * name<my_interfaces::msg::MultipleObstacles>()
{
  return "my_interfaces/msg/MultipleObstacles";
}

template<>
struct has_fixed_size<my_interfaces::msg::MultipleObstacles>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::msg::MultipleObstacles>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<my_interfaces::msg::MultipleObstacles>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__TRAITS_HPP_
