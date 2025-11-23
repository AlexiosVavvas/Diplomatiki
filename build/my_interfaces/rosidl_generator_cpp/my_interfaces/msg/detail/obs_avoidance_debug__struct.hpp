// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/ObsAvoidanceDebug.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__ObsAvoidanceDebug __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__ObsAvoidanceDebug __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ObsAvoidanceDebug_
{
  using Type = ObsAvoidanceDebug_<ContainerAllocator>;

  explicit ObsAvoidanceDebug_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->psi = 0.0;
      this->hddot = 0.0;
      this->two_alpha_h_hdot = 0.0;
      this->alpha2_h = 0.0;
    }
  }

  explicit ObsAvoidanceDebug_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->psi = 0.0;
      this->hddot = 0.0;
      this->two_alpha_h_hdot = 0.0;
      this->alpha2_h = 0.0;
    }
  }

  // field types and members
  using _psi_type =
    double;
  _psi_type psi;
  using _hddot_type =
    double;
  _hddot_type hddot;
  using _two_alpha_h_hdot_type =
    double;
  _two_alpha_h_hdot_type two_alpha_h_hdot;
  using _alpha2_h_type =
    double;
  _alpha2_h_type alpha2_h;
  using _beta_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _beta_type beta;
  using _u_safe_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _u_safe_type u_safe;

  // setters for named parameter idiom
  Type & set__psi(
    const double & _arg)
  {
    this->psi = _arg;
    return *this;
  }
  Type & set__hddot(
    const double & _arg)
  {
    this->hddot = _arg;
    return *this;
  }
  Type & set__two_alpha_h_hdot(
    const double & _arg)
  {
    this->two_alpha_h_hdot = _arg;
    return *this;
  }
  Type & set__alpha2_h(
    const double & _arg)
  {
    this->alpha2_h = _arg;
    return *this;
  }
  Type & set__beta(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->beta = _arg;
    return *this;
  }
  Type & set__u_safe(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->u_safe = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__ObsAvoidanceDebug
    std::shared_ptr<my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__ObsAvoidanceDebug
    std::shared_ptr<my_interfaces::msg::ObsAvoidanceDebug_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ObsAvoidanceDebug_ & other) const
  {
    if (this->psi != other.psi) {
      return false;
    }
    if (this->hddot != other.hddot) {
      return false;
    }
    if (this->two_alpha_h_hdot != other.two_alpha_h_hdot) {
      return false;
    }
    if (this->alpha2_h != other.alpha2_h) {
      return false;
    }
    if (this->beta != other.beta) {
      return false;
    }
    if (this->u_safe != other.u_safe) {
      return false;
    }
    return true;
  }
  bool operator!=(const ObsAvoidanceDebug_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ObsAvoidanceDebug_

// alias to use template instance with default allocator
using ObsAvoidanceDebug =
  my_interfaces::msg::ObsAvoidanceDebug_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__OBS_AVOIDANCE_DEBUG__STRUCT_HPP_
