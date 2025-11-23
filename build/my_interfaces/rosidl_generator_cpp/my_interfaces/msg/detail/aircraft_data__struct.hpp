// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/AircraftData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__AircraftData __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__AircraftData __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct AircraftData_
{
  using Type = AircraftData_<ContainerAllocator>;

  explicit AircraftData_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->north = 0.0;
      this->east = 0.0;
      this->down = 0.0;
      this->altitude = 0.0;
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
      this->roll_deg = 0.0;
      this->pitch_deg = 0.0;
      this->yaw_deg = 0.0;
      this->u_forward = 0.0;
      this->v_sideways = 0.0;
      this->w_downward = 0.0;
      this->airspeed = 0.0;
      this->v_north = 0.0;
      this->v_east = 0.0;
      this->v_down = 0.0;
      this->climb_rate = 0.0;
      this->ground_speed = 0.0;
      this->p_roll_rate = 0.0;
      this->q_pitch_rate = 0.0;
      this->r_yaw_rate = 0.0;
      this->p_deg_s = 0.0;
      this->q_deg_s = 0.0;
      this->r_deg_s = 0.0;
      this->alpha = 0.0;
      this->beta = 0.0;
      this->alpha_deg = 0.0;
      this->beta_deg = 0.0;
    }
  }

  explicit AircraftData_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->north = 0.0;
      this->east = 0.0;
      this->down = 0.0;
      this->altitude = 0.0;
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
      this->roll_deg = 0.0;
      this->pitch_deg = 0.0;
      this->yaw_deg = 0.0;
      this->u_forward = 0.0;
      this->v_sideways = 0.0;
      this->w_downward = 0.0;
      this->airspeed = 0.0;
      this->v_north = 0.0;
      this->v_east = 0.0;
      this->v_down = 0.0;
      this->climb_rate = 0.0;
      this->ground_speed = 0.0;
      this->p_roll_rate = 0.0;
      this->q_pitch_rate = 0.0;
      this->r_yaw_rate = 0.0;
      this->p_deg_s = 0.0;
      this->q_deg_s = 0.0;
      this->r_deg_s = 0.0;
      this->alpha = 0.0;
      this->beta = 0.0;
      this->alpha_deg = 0.0;
      this->beta_deg = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _north_type =
    double;
  _north_type north;
  using _east_type =
    double;
  _east_type east;
  using _down_type =
    double;
  _down_type down;
  using _altitude_type =
    double;
  _altitude_type altitude;
  using _roll_type =
    double;
  _roll_type roll;
  using _pitch_type =
    double;
  _pitch_type pitch;
  using _yaw_type =
    double;
  _yaw_type yaw;
  using _roll_deg_type =
    double;
  _roll_deg_type roll_deg;
  using _pitch_deg_type =
    double;
  _pitch_deg_type pitch_deg;
  using _yaw_deg_type =
    double;
  _yaw_deg_type yaw_deg;
  using _u_forward_type =
    double;
  _u_forward_type u_forward;
  using _v_sideways_type =
    double;
  _v_sideways_type v_sideways;
  using _w_downward_type =
    double;
  _w_downward_type w_downward;
  using _airspeed_type =
    double;
  _airspeed_type airspeed;
  using _v_north_type =
    double;
  _v_north_type v_north;
  using _v_east_type =
    double;
  _v_east_type v_east;
  using _v_down_type =
    double;
  _v_down_type v_down;
  using _climb_rate_type =
    double;
  _climb_rate_type climb_rate;
  using _ground_speed_type =
    double;
  _ground_speed_type ground_speed;
  using _p_roll_rate_type =
    double;
  _p_roll_rate_type p_roll_rate;
  using _q_pitch_rate_type =
    double;
  _q_pitch_rate_type q_pitch_rate;
  using _r_yaw_rate_type =
    double;
  _r_yaw_rate_type r_yaw_rate;
  using _p_deg_s_type =
    double;
  _p_deg_s_type p_deg_s;
  using _q_deg_s_type =
    double;
  _q_deg_s_type q_deg_s;
  using _r_deg_s_type =
    double;
  _r_deg_s_type r_deg_s;
  using _alpha_type =
    double;
  _alpha_type alpha;
  using _beta_type =
    double;
  _beta_type beta;
  using _alpha_deg_type =
    double;
  _alpha_deg_type alpha_deg;
  using _beta_deg_type =
    double;
  _beta_deg_type beta_deg;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__north(
    const double & _arg)
  {
    this->north = _arg;
    return *this;
  }
  Type & set__east(
    const double & _arg)
  {
    this->east = _arg;
    return *this;
  }
  Type & set__down(
    const double & _arg)
  {
    this->down = _arg;
    return *this;
  }
  Type & set__altitude(
    const double & _arg)
  {
    this->altitude = _arg;
    return *this;
  }
  Type & set__roll(
    const double & _arg)
  {
    this->roll = _arg;
    return *this;
  }
  Type & set__pitch(
    const double & _arg)
  {
    this->pitch = _arg;
    return *this;
  }
  Type & set__yaw(
    const double & _arg)
  {
    this->yaw = _arg;
    return *this;
  }
  Type & set__roll_deg(
    const double & _arg)
  {
    this->roll_deg = _arg;
    return *this;
  }
  Type & set__pitch_deg(
    const double & _arg)
  {
    this->pitch_deg = _arg;
    return *this;
  }
  Type & set__yaw_deg(
    const double & _arg)
  {
    this->yaw_deg = _arg;
    return *this;
  }
  Type & set__u_forward(
    const double & _arg)
  {
    this->u_forward = _arg;
    return *this;
  }
  Type & set__v_sideways(
    const double & _arg)
  {
    this->v_sideways = _arg;
    return *this;
  }
  Type & set__w_downward(
    const double & _arg)
  {
    this->w_downward = _arg;
    return *this;
  }
  Type & set__airspeed(
    const double & _arg)
  {
    this->airspeed = _arg;
    return *this;
  }
  Type & set__v_north(
    const double & _arg)
  {
    this->v_north = _arg;
    return *this;
  }
  Type & set__v_east(
    const double & _arg)
  {
    this->v_east = _arg;
    return *this;
  }
  Type & set__v_down(
    const double & _arg)
  {
    this->v_down = _arg;
    return *this;
  }
  Type & set__climb_rate(
    const double & _arg)
  {
    this->climb_rate = _arg;
    return *this;
  }
  Type & set__ground_speed(
    const double & _arg)
  {
    this->ground_speed = _arg;
    return *this;
  }
  Type & set__p_roll_rate(
    const double & _arg)
  {
    this->p_roll_rate = _arg;
    return *this;
  }
  Type & set__q_pitch_rate(
    const double & _arg)
  {
    this->q_pitch_rate = _arg;
    return *this;
  }
  Type & set__r_yaw_rate(
    const double & _arg)
  {
    this->r_yaw_rate = _arg;
    return *this;
  }
  Type & set__p_deg_s(
    const double & _arg)
  {
    this->p_deg_s = _arg;
    return *this;
  }
  Type & set__q_deg_s(
    const double & _arg)
  {
    this->q_deg_s = _arg;
    return *this;
  }
  Type & set__r_deg_s(
    const double & _arg)
  {
    this->r_deg_s = _arg;
    return *this;
  }
  Type & set__alpha(
    const double & _arg)
  {
    this->alpha = _arg;
    return *this;
  }
  Type & set__beta(
    const double & _arg)
  {
    this->beta = _arg;
    return *this;
  }
  Type & set__alpha_deg(
    const double & _arg)
  {
    this->alpha_deg = _arg;
    return *this;
  }
  Type & set__beta_deg(
    const double & _arg)
  {
    this->beta_deg = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::AircraftData_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::AircraftData_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::AircraftData_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::AircraftData_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::AircraftData_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::AircraftData_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::AircraftData_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::AircraftData_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::AircraftData_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::AircraftData_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__AircraftData
    std::shared_ptr<my_interfaces::msg::AircraftData_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__AircraftData
    std::shared_ptr<my_interfaces::msg::AircraftData_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AircraftData_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->north != other.north) {
      return false;
    }
    if (this->east != other.east) {
      return false;
    }
    if (this->down != other.down) {
      return false;
    }
    if (this->altitude != other.altitude) {
      return false;
    }
    if (this->roll != other.roll) {
      return false;
    }
    if (this->pitch != other.pitch) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    if (this->roll_deg != other.roll_deg) {
      return false;
    }
    if (this->pitch_deg != other.pitch_deg) {
      return false;
    }
    if (this->yaw_deg != other.yaw_deg) {
      return false;
    }
    if (this->u_forward != other.u_forward) {
      return false;
    }
    if (this->v_sideways != other.v_sideways) {
      return false;
    }
    if (this->w_downward != other.w_downward) {
      return false;
    }
    if (this->airspeed != other.airspeed) {
      return false;
    }
    if (this->v_north != other.v_north) {
      return false;
    }
    if (this->v_east != other.v_east) {
      return false;
    }
    if (this->v_down != other.v_down) {
      return false;
    }
    if (this->climb_rate != other.climb_rate) {
      return false;
    }
    if (this->ground_speed != other.ground_speed) {
      return false;
    }
    if (this->p_roll_rate != other.p_roll_rate) {
      return false;
    }
    if (this->q_pitch_rate != other.q_pitch_rate) {
      return false;
    }
    if (this->r_yaw_rate != other.r_yaw_rate) {
      return false;
    }
    if (this->p_deg_s != other.p_deg_s) {
      return false;
    }
    if (this->q_deg_s != other.q_deg_s) {
      return false;
    }
    if (this->r_deg_s != other.r_deg_s) {
      return false;
    }
    if (this->alpha != other.alpha) {
      return false;
    }
    if (this->beta != other.beta) {
      return false;
    }
    if (this->alpha_deg != other.alpha_deg) {
      return false;
    }
    if (this->beta_deg != other.beta_deg) {
      return false;
    }
    return true;
  }
  bool operator!=(const AircraftData_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AircraftData_

// alias to use template instance with default allocator
using AircraftData =
  my_interfaces::msg::AircraftData_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__AIRCRAFT_DATA__STRUCT_HPP_
