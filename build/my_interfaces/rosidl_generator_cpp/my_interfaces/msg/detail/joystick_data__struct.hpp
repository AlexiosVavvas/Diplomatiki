// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/JoystickData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__JoystickData __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__JoystickData __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct JoystickData_
{
  using Type = JoystickData_<ContainerAllocator>;

  explicit JoystickData_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->throttle = 0.0;
      this->aileron = 0.0;
      this->elevator = 0.0;
      this->rudder = 0.0;
      this->switch_state = 0l;
    }
  }

  explicit JoystickData_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->throttle = 0.0;
      this->aileron = 0.0;
      this->elevator = 0.0;
      this->rudder = 0.0;
      this->switch_state = 0l;
    }
  }

  // field types and members
  using _throttle_type =
    double;
  _throttle_type throttle;
  using _aileron_type =
    double;
  _aileron_type aileron;
  using _elevator_type =
    double;
  _elevator_type elevator;
  using _rudder_type =
    double;
  _rudder_type rudder;
  using _switch_state_type =
    int32_t;
  _switch_state_type switch_state;

  // setters for named parameter idiom
  Type & set__throttle(
    const double & _arg)
  {
    this->throttle = _arg;
    return *this;
  }
  Type & set__aileron(
    const double & _arg)
  {
    this->aileron = _arg;
    return *this;
  }
  Type & set__elevator(
    const double & _arg)
  {
    this->elevator = _arg;
    return *this;
  }
  Type & set__rudder(
    const double & _arg)
  {
    this->rudder = _arg;
    return *this;
  }
  Type & set__switch_state(
    const int32_t & _arg)
  {
    this->switch_state = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::JoystickData_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::JoystickData_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::JoystickData_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::JoystickData_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::JoystickData_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::JoystickData_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::JoystickData_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::JoystickData_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::JoystickData_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::JoystickData_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__JoystickData
    std::shared_ptr<my_interfaces::msg::JoystickData_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__JoystickData
    std::shared_ptr<my_interfaces::msg::JoystickData_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JoystickData_ & other) const
  {
    if (this->throttle != other.throttle) {
      return false;
    }
    if (this->aileron != other.aileron) {
      return false;
    }
    if (this->elevator != other.elevator) {
      return false;
    }
    if (this->rudder != other.rudder) {
      return false;
    }
    if (this->switch_state != other.switch_state) {
      return false;
    }
    return true;
  }
  bool operator!=(const JoystickData_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JoystickData_

// alias to use template instance with default allocator
using JoystickData =
  my_interfaces::msg::JoystickData_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__JOYSTICK_DATA__STRUCT_HPP_
