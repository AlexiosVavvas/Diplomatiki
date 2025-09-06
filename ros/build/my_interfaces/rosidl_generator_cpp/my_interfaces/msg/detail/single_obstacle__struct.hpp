// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/SingleObstacle.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__STRUCT_HPP_

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
# define DEPRECATED__my_interfaces__msg__SingleObstacle __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__SingleObstacle __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SingleObstacle_
{
  using Type = SingleObstacle_<ContainerAllocator>;

  explicit SingleObstacle_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->obs_type = "";
      this->obs_name = "";
      this->kappa = 0.0;
      this->rho0 = 0.0;
    }
  }

  explicit SingleObstacle_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : obs_type(_alloc),
    obs_name(_alloc),
    position(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->obs_type = "";
      this->obs_name = "";
      this->kappa = 0.0;
      this->rho0 = 0.0;
    }
  }

  // field types and members
  using _obs_type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _obs_type_type obs_type;
  using _obs_name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _obs_name_type obs_name;
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _dimensions_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _dimensions_type dimensions;
  using _kappa_type =
    double;
  _kappa_type kappa;
  using _rho0_type =
    double;
  _rho0_type rho0;

  // setters for named parameter idiom
  Type & set__obs_type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->obs_type = _arg;
    return *this;
  }
  Type & set__obs_name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->obs_name = _arg;
    return *this;
  }
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__dimensions(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->dimensions = _arg;
    return *this;
  }
  Type & set__kappa(
    const double & _arg)
  {
    this->kappa = _arg;
    return *this;
  }
  Type & set__rho0(
    const double & _arg)
  {
    this->rho0 = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::SingleObstacle_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::SingleObstacle_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::SingleObstacle_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::SingleObstacle_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::SingleObstacle_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::SingleObstacle_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::SingleObstacle_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::SingleObstacle_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::SingleObstacle_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::SingleObstacle_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__SingleObstacle
    std::shared_ptr<my_interfaces::msg::SingleObstacle_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__SingleObstacle
    std::shared_ptr<my_interfaces::msg::SingleObstacle_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SingleObstacle_ & other) const
  {
    if (this->obs_type != other.obs_type) {
      return false;
    }
    if (this->obs_name != other.obs_name) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->dimensions != other.dimensions) {
      return false;
    }
    if (this->kappa != other.kappa) {
      return false;
    }
    if (this->rho0 != other.rho0) {
      return false;
    }
    return true;
  }
  bool operator!=(const SingleObstacle_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SingleObstacle_

// alias to use template instance with default allocator
using SingleObstacle =
  my_interfaces::msg::SingleObstacle_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__SINGLE_OBSTACLE__STRUCT_HPP_
