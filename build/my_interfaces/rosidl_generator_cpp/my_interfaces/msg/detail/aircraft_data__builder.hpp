// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/AircraftData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/aircraft_data__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_AircraftData_beta_deg
{
public:
  explicit Init_AircraftData_beta_deg(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::AircraftData beta_deg(::my_interfaces::msg::AircraftData::_beta_deg_type arg)
  {
    msg_.beta_deg = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_alpha_deg
{
public:
  explicit Init_AircraftData_alpha_deg(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_beta_deg alpha_deg(::my_interfaces::msg::AircraftData::_alpha_deg_type arg)
  {
    msg_.alpha_deg = std::move(arg);
    return Init_AircraftData_beta_deg(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_beta
{
public:
  explicit Init_AircraftData_beta(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_alpha_deg beta(::my_interfaces::msg::AircraftData::_beta_type arg)
  {
    msg_.beta = std::move(arg);
    return Init_AircraftData_alpha_deg(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_alpha
{
public:
  explicit Init_AircraftData_alpha(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_beta alpha(::my_interfaces::msg::AircraftData::_alpha_type arg)
  {
    msg_.alpha = std::move(arg);
    return Init_AircraftData_beta(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_r_deg_s
{
public:
  explicit Init_AircraftData_r_deg_s(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_alpha r_deg_s(::my_interfaces::msg::AircraftData::_r_deg_s_type arg)
  {
    msg_.r_deg_s = std::move(arg);
    return Init_AircraftData_alpha(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_q_deg_s
{
public:
  explicit Init_AircraftData_q_deg_s(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_r_deg_s q_deg_s(::my_interfaces::msg::AircraftData::_q_deg_s_type arg)
  {
    msg_.q_deg_s = std::move(arg);
    return Init_AircraftData_r_deg_s(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_p_deg_s
{
public:
  explicit Init_AircraftData_p_deg_s(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_q_deg_s p_deg_s(::my_interfaces::msg::AircraftData::_p_deg_s_type arg)
  {
    msg_.p_deg_s = std::move(arg);
    return Init_AircraftData_q_deg_s(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_r_yaw_rate
{
public:
  explicit Init_AircraftData_r_yaw_rate(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_p_deg_s r_yaw_rate(::my_interfaces::msg::AircraftData::_r_yaw_rate_type arg)
  {
    msg_.r_yaw_rate = std::move(arg);
    return Init_AircraftData_p_deg_s(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_q_pitch_rate
{
public:
  explicit Init_AircraftData_q_pitch_rate(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_r_yaw_rate q_pitch_rate(::my_interfaces::msg::AircraftData::_q_pitch_rate_type arg)
  {
    msg_.q_pitch_rate = std::move(arg);
    return Init_AircraftData_r_yaw_rate(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_p_roll_rate
{
public:
  explicit Init_AircraftData_p_roll_rate(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_q_pitch_rate p_roll_rate(::my_interfaces::msg::AircraftData::_p_roll_rate_type arg)
  {
    msg_.p_roll_rate = std::move(arg);
    return Init_AircraftData_q_pitch_rate(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_ground_speed
{
public:
  explicit Init_AircraftData_ground_speed(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_p_roll_rate ground_speed(::my_interfaces::msg::AircraftData::_ground_speed_type arg)
  {
    msg_.ground_speed = std::move(arg);
    return Init_AircraftData_p_roll_rate(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_climb_rate
{
public:
  explicit Init_AircraftData_climb_rate(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_ground_speed climb_rate(::my_interfaces::msg::AircraftData::_climb_rate_type arg)
  {
    msg_.climb_rate = std::move(arg);
    return Init_AircraftData_ground_speed(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_v_down
{
public:
  explicit Init_AircraftData_v_down(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_climb_rate v_down(::my_interfaces::msg::AircraftData::_v_down_type arg)
  {
    msg_.v_down = std::move(arg);
    return Init_AircraftData_climb_rate(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_v_east
{
public:
  explicit Init_AircraftData_v_east(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_v_down v_east(::my_interfaces::msg::AircraftData::_v_east_type arg)
  {
    msg_.v_east = std::move(arg);
    return Init_AircraftData_v_down(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_v_north
{
public:
  explicit Init_AircraftData_v_north(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_v_east v_north(::my_interfaces::msg::AircraftData::_v_north_type arg)
  {
    msg_.v_north = std::move(arg);
    return Init_AircraftData_v_east(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_airspeed
{
public:
  explicit Init_AircraftData_airspeed(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_v_north airspeed(::my_interfaces::msg::AircraftData::_airspeed_type arg)
  {
    msg_.airspeed = std::move(arg);
    return Init_AircraftData_v_north(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_w_downward
{
public:
  explicit Init_AircraftData_w_downward(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_airspeed w_downward(::my_interfaces::msg::AircraftData::_w_downward_type arg)
  {
    msg_.w_downward = std::move(arg);
    return Init_AircraftData_airspeed(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_v_sideways
{
public:
  explicit Init_AircraftData_v_sideways(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_w_downward v_sideways(::my_interfaces::msg::AircraftData::_v_sideways_type arg)
  {
    msg_.v_sideways = std::move(arg);
    return Init_AircraftData_w_downward(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_u_forward
{
public:
  explicit Init_AircraftData_u_forward(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_v_sideways u_forward(::my_interfaces::msg::AircraftData::_u_forward_type arg)
  {
    msg_.u_forward = std::move(arg);
    return Init_AircraftData_v_sideways(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_yaw_deg
{
public:
  explicit Init_AircraftData_yaw_deg(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_u_forward yaw_deg(::my_interfaces::msg::AircraftData::_yaw_deg_type arg)
  {
    msg_.yaw_deg = std::move(arg);
    return Init_AircraftData_u_forward(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_pitch_deg
{
public:
  explicit Init_AircraftData_pitch_deg(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_yaw_deg pitch_deg(::my_interfaces::msg::AircraftData::_pitch_deg_type arg)
  {
    msg_.pitch_deg = std::move(arg);
    return Init_AircraftData_yaw_deg(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_roll_deg
{
public:
  explicit Init_AircraftData_roll_deg(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_pitch_deg roll_deg(::my_interfaces::msg::AircraftData::_roll_deg_type arg)
  {
    msg_.roll_deg = std::move(arg);
    return Init_AircraftData_pitch_deg(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_yaw
{
public:
  explicit Init_AircraftData_yaw(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_roll_deg yaw(::my_interfaces::msg::AircraftData::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_AircraftData_roll_deg(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_pitch
{
public:
  explicit Init_AircraftData_pitch(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_yaw pitch(::my_interfaces::msg::AircraftData::_pitch_type arg)
  {
    msg_.pitch = std::move(arg);
    return Init_AircraftData_yaw(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_roll
{
public:
  explicit Init_AircraftData_roll(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_pitch roll(::my_interfaces::msg::AircraftData::_roll_type arg)
  {
    msg_.roll = std::move(arg);
    return Init_AircraftData_pitch(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_altitude
{
public:
  explicit Init_AircraftData_altitude(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_roll altitude(::my_interfaces::msg::AircraftData::_altitude_type arg)
  {
    msg_.altitude = std::move(arg);
    return Init_AircraftData_roll(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_down
{
public:
  explicit Init_AircraftData_down(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_altitude down(::my_interfaces::msg::AircraftData::_down_type arg)
  {
    msg_.down = std::move(arg);
    return Init_AircraftData_altitude(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_east
{
public:
  explicit Init_AircraftData_east(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_down east(::my_interfaces::msg::AircraftData::_east_type arg)
  {
    msg_.east = std::move(arg);
    return Init_AircraftData_down(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_north
{
public:
  explicit Init_AircraftData_north(::my_interfaces::msg::AircraftData & msg)
  : msg_(msg)
  {}
  Init_AircraftData_east north(::my_interfaces::msg::AircraftData::_north_type arg)
  {
    msg_.north = std::move(arg);
    return Init_AircraftData_east(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

class Init_AircraftData_header
{
public:
  Init_AircraftData_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_AircraftData_north header(::my_interfaces::msg::AircraftData::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_AircraftData_north(msg_);
  }

private:
  ::my_interfaces::msg::AircraftData msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::AircraftData>()
{
  return my_interfaces::msg::builder::Init_AircraftData_header();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__BUILDER_HPP_
