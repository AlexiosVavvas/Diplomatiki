// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/ObsAvoidanceDebug.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/obs_avoidance_debug__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const ObsAvoidanceDebug & msg,
  std::ostream & out)
{
  out << "{";
  // member: psi
  {
    out << "psi: ";
    rosidl_generator_traits::value_to_yaml(msg.psi, out);
    out << ", ";
  }

  // member: hddot
  {
    out << "hddot: ";
    rosidl_generator_traits::value_to_yaml(msg.hddot, out);
    out << ", ";
  }

  // member: two_alpha_h_hdot
  {
    out << "two_alpha_h_hdot: ";
    rosidl_generator_traits::value_to_yaml(msg.two_alpha_h_hdot, out);
    out << ", ";
  }

  // member: alpha2_h
  {
    out << "alpha2_h: ";
    rosidl_generator_traits::value_to_yaml(msg.alpha2_h, out);
    out << ", ";
  }

  // member: beta
  {
    if (msg.beta.size() == 0) {
      out << "beta: []";
    } else {
      out << "beta: [";
      size_t pending_items = msg.beta.size();
      for (auto item : msg.beta) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: u_safe
  {
    if (msg.u_safe.size() == 0) {
      out << "u_safe: []";
    } else {
      out << "u_safe: [";
      size_t pending_items = msg.u_safe.size();
      for (auto item : msg.u_safe) {
        rosidl_generator_traits::value_to_yaml(item, out);
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
  const ObsAvoidanceDebug & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: psi
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "psi: ";
    rosidl_generator_traits::value_to_yaml(msg.psi, out);
    out << "\n";
  }

  // member: hddot
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "hddot: ";
    rosidl_generator_traits::value_to_yaml(msg.hddot, out);
    out << "\n";
  }

  // member: two_alpha_h_hdot
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "two_alpha_h_hdot: ";
    rosidl_generator_traits::value_to_yaml(msg.two_alpha_h_hdot, out);
    out << "\n";
  }

  // member: alpha2_h
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "alpha2_h: ";
    rosidl_generator_traits::value_to_yaml(msg.alpha2_h, out);
    out << "\n";
  }

  // member: beta
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.beta.size() == 0) {
      out << "beta: []\n";
    } else {
      out << "beta:\n";
      for (auto item : msg.beta) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: u_safe
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.u_safe.size() == 0) {
      out << "u_safe: []\n";
    } else {
      out << "u_safe:\n";
      for (auto item : msg.u_safe) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ObsAvoidanceDebug & msg, bool use_flow_style = false)
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
  const my_interfaces::msg::ObsAvoidanceDebug & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::ObsAvoidanceDebug & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::ObsAvoidanceDebug>()
{
  return "my_interfaces::msg::ObsAvoidanceDebug";
}

template<>
inline const char * name<my_interfaces::msg::ObsAvoidanceDebug>()
{
  return "my_interfaces/msg/ObsAvoidanceDebug";
}

template<>
struct has_fixed_size<my_interfaces::msg::ObsAvoidanceDebug>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::msg::ObsAvoidanceDebug>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<my_interfaces::msg::ObsAvoidanceDebug>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__TRAITS_HPP_
