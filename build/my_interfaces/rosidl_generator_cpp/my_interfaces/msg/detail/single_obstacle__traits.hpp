// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/SingleObstacle.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/single_obstacle__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const SingleObstacle & msg,
  std::ostream & out)
{
  out << "{";
  // member: obs_type
  {
    out << "obs_type: ";
    rosidl_generator_traits::value_to_yaml(msg.obs_type, out);
    out << ", ";
  }

  // member: obs_name
  {
    out << "obs_name: ";
    rosidl_generator_traits::value_to_yaml(msg.obs_name, out);
    out << ", ";
  }

  // member: position
  {
    out << "position: ";
    to_flow_style_yaml(msg.position, out);
    out << ", ";
  }

  // member: dimensions
  {
    if (msg.dimensions.size() == 0) {
      out << "dimensions: []";
    } else {
      out << "dimensions: [";
      size_t pending_items = msg.dimensions.size();
      for (auto item : msg.dimensions) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: kappa
  {
    out << "kappa: ";
    rosidl_generator_traits::value_to_yaml(msg.kappa, out);
    out << ", ";
  }

  // member: rho0
  {
    out << "rho0: ";
    rosidl_generator_traits::value_to_yaml(msg.rho0, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SingleObstacle & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: obs_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "obs_type: ";
    rosidl_generator_traits::value_to_yaml(msg.obs_type, out);
    out << "\n";
  }

  // member: obs_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "obs_name: ";
    rosidl_generator_traits::value_to_yaml(msg.obs_name, out);
    out << "\n";
  }

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position:\n";
    to_block_style_yaml(msg.position, out, indentation + 2);
  }

  // member: dimensions
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.dimensions.size() == 0) {
      out << "dimensions: []\n";
    } else {
      out << "dimensions:\n";
      for (auto item : msg.dimensions) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: kappa
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "kappa: ";
    rosidl_generator_traits::value_to_yaml(msg.kappa, out);
    out << "\n";
  }

  // member: rho0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rho0: ";
    rosidl_generator_traits::value_to_yaml(msg.rho0, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SingleObstacle & msg, bool use_flow_style = false)
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
  const my_interfaces::msg::SingleObstacle & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::SingleObstacle & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::SingleObstacle>()
{
  return "my_interfaces::msg::SingleObstacle";
}

template<>
inline const char * name<my_interfaces::msg::SingleObstacle>()
{
  return "my_interfaces/msg/SingleObstacle";
}

template<>
struct has_fixed_size<my_interfaces::msg::SingleObstacle>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::msg::SingleObstacle>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<my_interfaces::msg::SingleObstacle>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__TRAITS_HPP_
