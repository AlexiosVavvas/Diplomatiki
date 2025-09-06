// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/AgentData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__AGENT_DATA__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__AGENT_DATA__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/agent_data__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const AgentData & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: simulation_time
  {
    out << "simulation_time: ";
    rosidl_generator_traits::value_to_yaml(msg.simulation_time, out);
    out << ", ";
  }

  // member: num_of_states
  {
    out << "num_of_states: ";
    rosidl_generator_traits::value_to_yaml(msg.num_of_states, out);
    out << ", ";
  }

  // member: num_of_inputs
  {
    out << "num_of_inputs: ";
    rosidl_generator_traits::value_to_yaml(msg.num_of_inputs, out);
    out << ", ";
  }

  // member: states
  {
    if (msg.states.size() == 0) {
      out << "states: []";
    } else {
      out << "states: [";
      size_t pending_items = msg.states.size();
      for (auto item : msg.states) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: inputs
  {
    if (msg.inputs.size() == 0) {
      out << "inputs: []";
    } else {
      out << "inputs: [";
      size_t pending_items = msg.inputs.size();
      for (auto item : msg.inputs) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ergodic_cost
  {
    out << "ergodic_cost: ";
    rosidl_generator_traits::value_to_yaml(msg.ergodic_cost, out);
    out << ", ";
  }

  // member: active_cbf_flag
  {
    out << "active_cbf_flag: ";
    rosidl_generator_traits::value_to_yaml(msg.active_cbf_flag, out);
    out << ", ";
  }

  // member: in_range_agents_ids
  {
    if (msg.in_range_agents_ids.size() == 0) {
      out << "in_range_agents_ids: []";
    } else {
      out << "in_range_agents_ids: [";
      size_t pending_items = msg.in_range_agents_ids.size();
      for (auto item : msg.in_range_agents_ids) {
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
  const AgentData & msg,
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

  // member: simulation_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "simulation_time: ";
    rosidl_generator_traits::value_to_yaml(msg.simulation_time, out);
    out << "\n";
  }

  // member: num_of_states
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_of_states: ";
    rosidl_generator_traits::value_to_yaml(msg.num_of_states, out);
    out << "\n";
  }

  // member: num_of_inputs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_of_inputs: ";
    rosidl_generator_traits::value_to_yaml(msg.num_of_inputs, out);
    out << "\n";
  }

  // member: states
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.states.size() == 0) {
      out << "states: []\n";
    } else {
      out << "states:\n";
      for (auto item : msg.states) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: inputs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.inputs.size() == 0) {
      out << "inputs: []\n";
    } else {
      out << "inputs:\n";
      for (auto item : msg.inputs) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: ergodic_cost
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ergodic_cost: ";
    rosidl_generator_traits::value_to_yaml(msg.ergodic_cost, out);
    out << "\n";
  }

  // member: active_cbf_flag
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "active_cbf_flag: ";
    rosidl_generator_traits::value_to_yaml(msg.active_cbf_flag, out);
    out << "\n";
  }

  // member: in_range_agents_ids
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.in_range_agents_ids.size() == 0) {
      out << "in_range_agents_ids: []\n";
    } else {
      out << "in_range_agents_ids:\n";
      for (auto item : msg.in_range_agents_ids) {
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

inline std::string to_yaml(const AgentData & msg, bool use_flow_style = false)
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
  const my_interfaces::msg::AgentData & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::AgentData & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::AgentData>()
{
  return "my_interfaces::msg::AgentData";
}

template<>
inline const char * name<my_interfaces::msg::AgentData>()
{
  return "my_interfaces/msg/AgentData";
}

template<>
struct has_fixed_size<my_interfaces::msg::AgentData>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::msg::AgentData>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<my_interfaces::msg::AgentData>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__AGENT_DATA__TRAITS_HPP_
