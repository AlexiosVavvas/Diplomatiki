// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/AircraftData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/aircraft_data__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const AircraftData & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: north
  {
    out << "north: ";
    rosidl_generator_traits::value_to_yaml(msg.north, out);
    out << ", ";
  }

  // member: east
  {
    out << "east: ";
    rosidl_generator_traits::value_to_yaml(msg.east, out);
    out << ", ";
  }

  // member: down
  {
    out << "down: ";
    rosidl_generator_traits::value_to_yaml(msg.down, out);
    out << ", ";
  }

  // member: altitude
  {
    out << "altitude: ";
    rosidl_generator_traits::value_to_yaml(msg.altitude, out);
    out << ", ";
  }

  // member: roll
  {
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << ", ";
  }

  // member: pitch
  {
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << ", ";
  }

  // member: yaw
  {
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
    out << ", ";
  }

  // member: roll_deg
  {
    out << "roll_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.roll_deg, out);
    out << ", ";
  }

  // member: pitch_deg
  {
    out << "pitch_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch_deg, out);
    out << ", ";
  }

  // member: yaw_deg
  {
    out << "yaw_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_deg, out);
    out << ", ";
  }

  // member: u_forward
  {
    out << "u_forward: ";
    rosidl_generator_traits::value_to_yaml(msg.u_forward, out);
    out << ", ";
  }

  // member: v_sideways
  {
    out << "v_sideways: ";
    rosidl_generator_traits::value_to_yaml(msg.v_sideways, out);
    out << ", ";
  }

  // member: w_downward
  {
    out << "w_downward: ";
    rosidl_generator_traits::value_to_yaml(msg.w_downward, out);
    out << ", ";
  }

  // member: airspeed
  {
    out << "airspeed: ";
    rosidl_generator_traits::value_to_yaml(msg.airspeed, out);
    out << ", ";
  }

  // member: v_north
  {
    out << "v_north: ";
    rosidl_generator_traits::value_to_yaml(msg.v_north, out);
    out << ", ";
  }

  // member: v_east
  {
    out << "v_east: ";
    rosidl_generator_traits::value_to_yaml(msg.v_east, out);
    out << ", ";
  }

  // member: v_down
  {
    out << "v_down: ";
    rosidl_generator_traits::value_to_yaml(msg.v_down, out);
    out << ", ";
  }

  // member: climb_rate
  {
    out << "climb_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.climb_rate, out);
    out << ", ";
  }

  // member: ground_speed
  {
    out << "ground_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.ground_speed, out);
    out << ", ";
  }

  // member: p_roll_rate
  {
    out << "p_roll_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.p_roll_rate, out);
    out << ", ";
  }

  // member: q_pitch_rate
  {
    out << "q_pitch_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.q_pitch_rate, out);
    out << ", ";
  }

  // member: r_yaw_rate
  {
    out << "r_yaw_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.r_yaw_rate, out);
    out << ", ";
  }

  // member: p_deg_s
  {
    out << "p_deg_s: ";
    rosidl_generator_traits::value_to_yaml(msg.p_deg_s, out);
    out << ", ";
  }

  // member: q_deg_s
  {
    out << "q_deg_s: ";
    rosidl_generator_traits::value_to_yaml(msg.q_deg_s, out);
    out << ", ";
  }

  // member: r_deg_s
  {
    out << "r_deg_s: ";
    rosidl_generator_traits::value_to_yaml(msg.r_deg_s, out);
    out << ", ";
  }

  // member: alpha
  {
    out << "alpha: ";
    rosidl_generator_traits::value_to_yaml(msg.alpha, out);
    out << ", ";
  }

  // member: beta
  {
    out << "beta: ";
    rosidl_generator_traits::value_to_yaml(msg.beta, out);
    out << ", ";
  }

  // member: alpha_deg
  {
    out << "alpha_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.alpha_deg, out);
    out << ", ";
  }

  // member: beta_deg
  {
    out << "beta_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.beta_deg, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const AircraftData & msg,
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

  // member: north
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "north: ";
    rosidl_generator_traits::value_to_yaml(msg.north, out);
    out << "\n";
  }

  // member: east
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "east: ";
    rosidl_generator_traits::value_to_yaml(msg.east, out);
    out << "\n";
  }

  // member: down
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "down: ";
    rosidl_generator_traits::value_to_yaml(msg.down, out);
    out << "\n";
  }

  // member: altitude
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "altitude: ";
    rosidl_generator_traits::value_to_yaml(msg.altitude, out);
    out << "\n";
  }

  // member: roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << "\n";
  }

  // member: pitch
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << "\n";
  }

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
    out << "\n";
  }

  // member: roll_deg
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.roll_deg, out);
    out << "\n";
  }

  // member: pitch_deg
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pitch_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch_deg, out);
    out << "\n";
  }

  // member: yaw_deg
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_deg, out);
    out << "\n";
  }

  // member: u_forward
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "u_forward: ";
    rosidl_generator_traits::value_to_yaml(msg.u_forward, out);
    out << "\n";
  }

  // member: v_sideways
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_sideways: ";
    rosidl_generator_traits::value_to_yaml(msg.v_sideways, out);
    out << "\n";
  }

  // member: w_downward
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "w_downward: ";
    rosidl_generator_traits::value_to_yaml(msg.w_downward, out);
    out << "\n";
  }

  // member: airspeed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "airspeed: ";
    rosidl_generator_traits::value_to_yaml(msg.airspeed, out);
    out << "\n";
  }

  // member: v_north
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_north: ";
    rosidl_generator_traits::value_to_yaml(msg.v_north, out);
    out << "\n";
  }

  // member: v_east
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_east: ";
    rosidl_generator_traits::value_to_yaml(msg.v_east, out);
    out << "\n";
  }

  // member: v_down
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_down: ";
    rosidl_generator_traits::value_to_yaml(msg.v_down, out);
    out << "\n";
  }

  // member: climb_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "climb_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.climb_rate, out);
    out << "\n";
  }

  // member: ground_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ground_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.ground_speed, out);
    out << "\n";
  }

  // member: p_roll_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "p_roll_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.p_roll_rate, out);
    out << "\n";
  }

  // member: q_pitch_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "q_pitch_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.q_pitch_rate, out);
    out << "\n";
  }

  // member: r_yaw_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "r_yaw_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.r_yaw_rate, out);
    out << "\n";
  }

  // member: p_deg_s
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "p_deg_s: ";
    rosidl_generator_traits::value_to_yaml(msg.p_deg_s, out);
    out << "\n";
  }

  // member: q_deg_s
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "q_deg_s: ";
    rosidl_generator_traits::value_to_yaml(msg.q_deg_s, out);
    out << "\n";
  }

  // member: r_deg_s
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "r_deg_s: ";
    rosidl_generator_traits::value_to_yaml(msg.r_deg_s, out);
    out << "\n";
  }

  // member: alpha
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "alpha: ";
    rosidl_generator_traits::value_to_yaml(msg.alpha, out);
    out << "\n";
  }

  // member: beta
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "beta: ";
    rosidl_generator_traits::value_to_yaml(msg.beta, out);
    out << "\n";
  }

  // member: alpha_deg
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "alpha_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.alpha_deg, out);
    out << "\n";
  }

  // member: beta_deg
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "beta_deg: ";
    rosidl_generator_traits::value_to_yaml(msg.beta_deg, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const AircraftData & msg, bool use_flow_style = false)
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
  const my_interfaces::msg::AircraftData & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::AircraftData & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::AircraftData>()
{
  return "my_interfaces::msg::AircraftData";
}

template<>
inline const char * name<my_interfaces::msg::AircraftData>()
{
  return "my_interfaces/msg/AircraftData";
}

template<>
struct has_fixed_size<my_interfaces::msg::AircraftData>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<my_interfaces::msg::AircraftData>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<my_interfaces::msg::AircraftData>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__TRAITS_HPP_
