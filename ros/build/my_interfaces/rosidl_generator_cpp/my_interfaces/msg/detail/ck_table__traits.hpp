// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/CkTable.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__CK_TABLE__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__CK_TABLE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/ck_table__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const CkTable & msg,
  std::ostream & out)
{
  out << "{";
  // member: table_size
  {
    out << "table_size: ";
    rosidl_generator_traits::value_to_yaml(msg.table_size, out);
    out << ", ";
  }

  // member: ck_values
  {
    if (msg.ck_values.size() == 0) {
      out << "ck_values: []";
    } else {
      out << "ck_values: [";
      size_t pending_items = msg.ck_values.size();
      for (auto item : msg.ck_values) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: total_erg_cost
  {
    out << "total_erg_cost: ";
    rosidl_generator_traits::value_to_yaml(msg.total_erg_cost, out);
    out << ", ";
  }

  // member: total_erg_cost_in_range
  {
    out << "total_erg_cost_in_range: ";
    rosidl_generator_traits::value_to_yaml(msg.total_erg_cost_in_range, out);
    out << ", ";
  }

  // member: position
  {
    out << "position: ";
    to_flow_style_yaml(msg.position, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CkTable & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: table_size
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "table_size: ";
    rosidl_generator_traits::value_to_yaml(msg.table_size, out);
    out << "\n";
  }

  // member: ck_values
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ck_values.size() == 0) {
      out << "ck_values: []\n";
    } else {
      out << "ck_values:\n";
      for (auto item : msg.ck_values) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: total_erg_cost
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "total_erg_cost: ";
    rosidl_generator_traits::value_to_yaml(msg.total_erg_cost, out);
    out << "\n";
  }

  // member: total_erg_cost_in_range
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "total_erg_cost_in_range: ";
    rosidl_generator_traits::value_to_yaml(msg.total_erg_cost_in_range, out);
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
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CkTable & msg, bool use_flow_style = false)
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
  const my_interfaces::msg::CkTable & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::CkTable & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::CkTable>()
{
  return "my_interfaces::msg::CkTable";
}

template<>
inline const char * name<my_interfaces::msg::CkTable>()
{
  return "my_interfaces/msg/CkTable";
}

template<>
struct has_fixed_size<my_interfaces::msg::CkTable>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::msg::CkTable>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<my_interfaces::msg::CkTable>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__CK_TABLE__TRAITS_HPP_
