// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/MultipleTargetEstimates.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__STRUCT_HPP_

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
// Member 'target_estimates'
// Member 'ground_truths'
#include "my_interfaces/msg/detail/single_target_estimate__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__MultipleTargetEstimates __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__MultipleTargetEstimates __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MultipleTargetEstimates_
{
  using Type = MultipleTargetEstimates_<ContainerAllocator>;

  explicit MultipleTargetEstimates_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_of_targets = 0l;
    }
  }

  explicit MultipleTargetEstimates_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_of_targets = 0l;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _num_of_targets_type =
    int32_t;
  _num_of_targets_type num_of_targets;
  using _target_estimates_type =
    std::vector<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>>;
  _target_estimates_type target_estimates;
  using _ground_truths_type =
    std::vector<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>>;
  _ground_truths_type ground_truths;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__num_of_targets(
    const int32_t & _arg)
  {
    this->num_of_targets = _arg;
    return *this;
  }
  Type & set__target_estimates(
    const std::vector<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>> & _arg)
  {
    this->target_estimates = _arg;
    return *this;
  }
  Type & set__ground_truths(
    const std::vector<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<my_interfaces::msg::SingleTargetEstimate_<ContainerAllocator>>> & _arg)
  {
    this->ground_truths = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__MultipleTargetEstimates
    std::shared_ptr<my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__MultipleTargetEstimates
    std::shared_ptr<my_interfaces::msg::MultipleTargetEstimates_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MultipleTargetEstimates_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->num_of_targets != other.num_of_targets) {
      return false;
    }
    if (this->target_estimates != other.target_estimates) {
      return false;
    }
    if (this->ground_truths != other.ground_truths) {
      return false;
    }
    return true;
  }
  bool operator!=(const MultipleTargetEstimates_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MultipleTargetEstimates_

// alias to use template instance with default allocator
using MultipleTargetEstimates =
  my_interfaces::msg::MultipleTargetEstimates_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__MULTIPLE_TARGET_ESTIMATES__STRUCT_HPP_
