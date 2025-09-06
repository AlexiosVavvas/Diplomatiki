// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/MultipleObstacles.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'obstacles'
#include "my_interfaces/msg/detail/single_obstacle__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__MultipleObstacles __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__MultipleObstacles __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MultipleObstacles_
{
  using Type = MultipleObstacles_<ContainerAllocator>;

  explicit MultipleObstacles_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_of_obstacles = 0l;
    }
  }

  explicit MultipleObstacles_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->num_of_obstacles = 0l;
    }
  }

  // field types and members
  using _num_of_obstacles_type =
    int32_t;
  _num_of_obstacles_type num_of_obstacles;
  using _obstacles_type =
    std::vector<my_interfaces::msg::SingleObstacle_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<my_interfaces::msg::SingleObstacle_<ContainerAllocator>>>;
  _obstacles_type obstacles;

  // setters for named parameter idiom
  Type & set__num_of_obstacles(
    const int32_t & _arg)
  {
    this->num_of_obstacles = _arg;
    return *this;
  }
  Type & set__obstacles(
    const std::vector<my_interfaces::msg::SingleObstacle_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<my_interfaces::msg::SingleObstacle_<ContainerAllocator>>> & _arg)
  {
    this->obstacles = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::MultipleObstacles_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::MultipleObstacles_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::MultipleObstacles_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::MultipleObstacles_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::MultipleObstacles_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::MultipleObstacles_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::MultipleObstacles_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::MultipleObstacles_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::MultipleObstacles_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::MultipleObstacles_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__MultipleObstacles
    std::shared_ptr<my_interfaces::msg::MultipleObstacles_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__MultipleObstacles
    std::shared_ptr<my_interfaces::msg::MultipleObstacles_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MultipleObstacles_ & other) const
  {
    if (this->num_of_obstacles != other.num_of_obstacles) {
      return false;
    }
    if (this->obstacles != other.obstacles) {
      return false;
    }
    return true;
  }
  bool operator!=(const MultipleObstacles_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MultipleObstacles_

// alias to use template instance with default allocator
using MultipleObstacles =
  my_interfaces::msg::MultipleObstacles_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__STRUCT_HPP_
