// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/SingleTargetEstimate.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__SingleTargetEstimate __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__SingleTargetEstimate __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SingleTargetEstimate_
{
  using Type = SingleTargetEstimate_<ContainerAllocator>;

  explicit SingleTargetEstimate_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::DEFAULTS_ONLY == _init)
    {
      this->target_id = -1l;
      std::fill<typename std::array<double, 9>::iterator, double>(this->covariance.begin(), this->covariance.end(), 0.0);
    } else if (rosidl_runtime_cpp::MessageInitialization::ZERO == _init) {
      this->target_id = 0l;
      std::fill<typename std::array<double, 9>::iterator, double>(this->covariance.begin(), this->covariance.end(), 0.0);
    }
  }

  explicit SingleTargetEstimate_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_alloc, _init),
    covariance(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::DEFAULTS_ONLY == _init)
    {
      this->target_id = -1l;
      std::fill<typename std::array<double, 9>::iterator, double>(this->covariance.begin(), this->covariance.end(), 0.0);
    } else if (rosidl_runtime_cpp::MessageInitialization::ZERO == _init) {
      this->target_id = 0l;
      std::fill<typename std::array<double, 9>::iterator, double>(this->covariance.begin(), this->covariance.end(), 0.0);
    }
  }

  // field types and members
  using _target_id_type =
    int32_t;
  _target_id_type target_id;
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _covariance_type =
    std::array<double, 9>;
  _covariance_type covariance;

  // setters for named parameter idiom
  Type & set__target_id(
    const int32_t & _arg)
  {
    this->target_id = _arg;
    return *this;
  }
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__covariance(
    const std::array<double, 9> & _arg)
  {
    this->covariance = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__SingleTargetEstimate
    std::shared_ptr<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__SingleTargetEstimate
    std::shared_ptr<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SingleTargetEstimate_ & other) const
  {
    if (this->target_id != other.target_id) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->covariance != other.covariance) {
      return false;
    }
    return true;
  }
  bool operator!=(const SingleTargetEstimate_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SingleTargetEstimate_

// alias to use template instance with default allocator
using SingleTargetEstimate =
  my_interfaces::msg::SingleTargetEstimate_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__SINGLE_TARGET_ESTIMATE__STRUCT_HPP_
