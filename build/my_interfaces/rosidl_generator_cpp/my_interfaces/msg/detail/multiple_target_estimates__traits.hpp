// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/MultipleTargetEstimates.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/multiple_target_estimates__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'target_estimates'
// Member 'ground_truths'
#include "my_interfaces/msg/detail/single_target_estimate__traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const MultipleTargetEstimates & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: num_of_targets
  {
    out << "num_of_targets: ";
    rosidl_generator_traits::value_to_yaml(msg.num_of_targets, out);
    out << ", ";
  }

  // member: target_estimates
  {
    if (msg.target_estimates.size() == 0) {
      out << "target_estimates: []";
    } else {
      out << "target_estimates: [";
      size_t pending_items = msg.target_estimates.size();
      for (auto item : msg.target_estimates) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ground_truths
  {
    if (msg.ground_truths.size() == 0) {
      out << "ground_truths: []";
    } else {
      out << "ground_truths: [";
      size_t pending_items = msg.ground_truths.size();
      for (auto item : msg.ground_truths) {
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
  const MultipleTargetEstimates & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: num_of_targets
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_of_targets: ";
    rosidl_generator_traits::value_to_yaml(msg.num_of_targets, out);
    out << "\n";
  }

  // member: target_estimates
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.target_estimates.size() == 0) {
      out << "target_estimates: []\n";
    } else {
      out << "target_estimates:\n";
      for (auto item : msg.target_estimates) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: ground_truths
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ground_truths.size() == 0) {
      out << "ground_truths: []\n";
    } else {
      out << "ground_truths:\n";
      for (auto item : msg.ground_truths) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MultipleTargetEstimates & msg, bool use_flow_style = false)
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
  const my_interfaces::msg::MultipleTargetEstimates & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::MultipleTargetEstimates & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::MultipleTargetEstimates>()
{
  return "my_interfaces::msg::MultipleTargetEstimates";
}

template<>
inline const char * name<my_interfaces::msg::MultipleTargetEstimates>()
{
  return "my_interfaces/msg/MultipleTargetEstimates";
}

template<>
struct has_fixed_size<my_interfaces::msg::MultipleTargetEstimates>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::msg::MultipleTargetEstimates>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<my_interfaces::msg::MultipleTargetEstimates>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__TRAITS_HPP_
